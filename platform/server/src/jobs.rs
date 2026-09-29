//! 任务引擎:把 rsface core 当 SDK 用,驱动 图片/视频/直播流 三类任务。
//!
//! 数据流:
//! ```text
//!   upload / URL ──> 临时文件 ──> source::open() ──> Frame 循环
//!        │                                              │
//!        │                                   Detector::detect(gray)
//!        │                                              │
//!        ▼                                     标注帧 PNG + 人脸裁剪 PNG
//!   存储: jobs/{id}/original.* | annotated/ | frames/ | faces/
//!        │       (S3 优先,失败降级本地)
//!        ▼
//!   JobRegistry(内存索引)+ SSE 实时事件 ──> Web 前端
//! ```
//!
//! 稳定性 / 并发要点:
//! - `JobRegistry` 通过 `tokio::sync::Semaphore` 限制同时跑的 job 数;
//! - 每个 job 跑在 `std::thread::spawn` 内,顶层用 `std::panic::catch_unwind`
//!   兜住 panic,转 `JobStatus::Error`,避免拖垮 axum runtime;
//! - 每个 job 启动一个 watchdog 线程,超时则 `cancel.store(true)`;
//! - 任意 job 的 S3+local 双写都失败时,把 base64 内嵌到 SSE `frame` 事件,
//!   前端可以直接渲染(优雅降级);
//! - SSE 每 15 秒发注释 `:keepalive` 帧,防中间代理切断。

use crate::config::Config;
use crate::metrics::PlatformMetrics;
use crate::persist::Db;
use crate::s3::S3Client;
use rsface::cnn::{CnnConfig, CnnDetector, CnnWeights};
use rsface::detector::{Detection, Detector, DetectorConfig};
use rsface::face_detector::FaceDetector;
use rsface::haar::Cascade;
use rsface::hog_face::{HogConfig, HogFaceDetector};
use rsface::image::{png::write_png_rgb, GrayImage, RgbImage};
use rsface::lbp_face::{LbpConfig, LbpFaceDetector};
use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};
use rsface::skin_face::{SkinConfig, SkinFaceDetector};
use rsface::source::{open as open_source, Frame};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, Semaphore};

#[derive(Clone, Copy, PartialEq, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Image,
    Video,
    Stream,
}

#[derive(Clone, Copy, PartialEq, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Done,
    Cancelled,
    Error,
}

#[derive(Clone, Serialize, Debug)]
pub struct FaceEntry {
    /// S3 key of the cropped face PNG.
    pub key: String,
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub score: f32,
    /// Silent liveness verdict, present only when liveness is enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub liveness: Option<crate::liveness::LivenessVerdict>,
    /// True when liveness enforcement blocks this detection.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub blocked: bool,
}

#[derive(Clone, Serialize, Debug)]
pub struct FrameResult {
    pub index: u64,
    pub timestamp_ms: u64,
    /// S3 key of the annotated (boxes drawn) frame PNG; set when the frame has faces
    /// or is a keepalive sample for live streams.
    pub annotated_key: Option<String>,
    /// S3 key of the untouched frame PNG (live streams only, for compare view).
    pub original_key: Option<String>,
    pub faces: Vec<FaceEntry>,
}

#[derive(Clone, Serialize, Debug, Default)]
pub struct JobStats {
    pub frames_processed: u64,
    /// 视频总帧数(视频任务;来自帧源 total_hint,ffprobe)。
    /// 0 = 未知(直播流 / ffprobe 不可用),前端进度条降级显示处理帧数。
    pub total_frames: u64,
    pub frames_with_face: u64,
    pub total_detections: u64,
    /// Detections judged non-real by liveness (spoof_detections).
    pub spoof_detections: u64,
    /// Detections blocked under liveness enforcement.
    pub blocked_detections: u64,
    pub elapsed_ms: u64,
    /// 当前 job 使用的算法(haar/cnn/luminance;历史数据里也可能出现已下线的
    /// yunet/mtcnn/hog,展示与聚合时原样保留)。
    /// run_job 构建 detector 后写入,供 stats 聚合 + 前端过滤 chip 使用。
    pub algo: String,
}

pub struct Job {
    pub id: String,
    pub kind: JobKind,
    /// 展示名:上传文件名或流地址。
    pub display_name: String,
    pub created_ms: u64,
    pub status: Mutex<JobStatus>,
    pub frames: Mutex<Vec<FrameResult>>,
    pub stats: Mutex<JobStats>,
    /// 检测算法名(haar/cnn/luminance),由 run_job 在 detector 构建后写入,
    /// 摘要里暴露给前端做算法过滤 chip。
    pub algo: Mutex<Option<String>>,
    /// 用户在 upload 时指定的算法(haar/cnn/luminance);
    /// 优先于 `RSFACE_ALGO` 环境变量。`None` = 走环境变量 / 默认。
    /// 在 `Job::create` 之前由 `api::handle_upload` 写入(基于 multipart
    /// `algo` 字段),`run_job` 在构建 detector 时读取。
    pub algo_override: Mutex<Option<String>>,
    /// 原始媒体 S3 key(图片/视频任务)。
    pub original_media_key: Mutex<Option<String>>,
    pub error: Mutex<Option<String>>,
    /// 归档标志:归档后侧栏默认隐藏(但仍在内存/DB 中)。
    pub archived: Mutex<bool>,
    /// 原始输入(URL 或 upload 文件名),供"重试"用。
    pub original_input: Mutex<Option<String>>,
    /// 用 `Arc` 包裹以便 watchdog 线程独立持有。
    pub cancel: Arc<AtomicBool>,
    /// SSE 事件通道(payload 为 JSON 字符串)。
    pub event_tx: broadcast::Sender<String>,
    /// 任务运行线程的 JoinHandle。中 #5:cleanup 删除媒体前 join 一下,
    /// 防止后台线程在 delete 之后继续写 S3 对象(产生孤儿)。
    pub worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// `/api/jobs/{id}/recognize` 的 per-job 缓存。
    /// 键:`{job_id}|{detector}|{media_key}`;失效:media_key 变化(用户
    /// 重新上传)→ 旧 key 不命中,自动重建。
    /// 锁粒度:识别 handler 只短持锁 clone body,不会拖住 spawn_blocking。
    pub recognize_cache: Mutex<Option<RecognizeCache>>,
    /// 弱引用回 registry,仅 `set_status` 用来 bump 聚合计数。
    /// 弱引用是为了不形成 `Arc<Registry> → HashMap<_, Arc<Job>> → Job →
    /// Arc<Registry>` 的环。Registry 在 set_status 时若已被 drop,
    /// `upgrade()` 返回 None,直接跳过即可,不影响 job 状态。
    pub registry: std::sync::Weak<JobRegistry>,
}

/// `/recognize` per-job 缓存实体。`body` 是可直接返回给前端的 JSON Value,
/// 命中缓存时跳过检测 + 共识融合(每次 ~5-20 ms 单图)。
#[derive(Clone, Debug)]
pub struct RecognizeCache {
    pub cache_key: String,
    pub body: serde_json::Value,
}

impl Job {
    /// poison 容忍:持锁线程 panic 时仍读出最后状态。跨 job 聚合循环
    /// (metrics / list)会遍历所有 job,单个死 job 不应 panic 整个端点。
    pub fn status(&self) -> JobStatus {
        *self.status.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn set_status(&self, s: JobStatus) {
        // 老实现裸写,这里加一层:把状态变化推到 registry 聚合计数。
        // 调用方原本就只调 `set_status` 一处,所以取旧值 / 写新值 / 推计数
        // 都在这里,语义对外不变。
        let prev = *self.status.lock().unwrap_or_else(|e| e.into_inner());
        if prev == s {
            return;
        }
        *self.status.lock().unwrap_or_else(|e| e.into_inner()) = s;
        if let Some(reg) = self.registry.upgrade() {
            reg.transition_status_counter(prev, s);
        }
    }

    pub fn emit(&self, payload: &str) {
        // 无订阅者时忽略发送错误。
        let _ = self.event_tx.send(payload.to_string());
    }

    /// face 总数。注意:调用方若已持有 `frames` 锁,必须改用
    /// `face_count_locked(&frames)`,std::sync::Mutex 不可重入,
    /// 二次 lock 会死锁(修复:原 summary() 在持有 frames 锁时调
    /// self.face_count(),GET /api/jobs 会永久挂起)。
    #[allow(dead_code)] // 公共访问器;summary 内部走 face_count_locked 避免重入
    pub fn face_count(&self) -> usize {
        face_count_locked(&self.frames.lock().unwrap_or_else(|e| e.into_inner()))
    }

    pub fn summary(&self) -> serde_json::Value {
        // For video/stream jobs, prefer the first annotated frame as the cover
        // thumbnail (the original key points to .mp4 which renders as a broken
        // image in <img>). For image jobs, the original_key IS the cover.
        let frames = self.frames.lock().unwrap_or_else(|e| e.into_inner());
        let face_count = face_count_locked(&frames);
        let cover_key: Option<String> = match self.kind {
            JobKind::Image => self
                .original_media_key
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            _ => frames
                .iter()
                .find_map(|f| f.annotated_key.clone())
                .or_else(|| {
                    self.original_media_key
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone()
                }),
        };
        let frame_count = frames.len();
        drop(frames);
        serde_json::json!({
            "id": self.id,
            "kind": self.kind,
            "display_name": self.display_name,
            "status": self.status(),
            "created_ms": self.created_ms,
            "stats": *self.stats.lock().unwrap_or_else(|e| e.into_inner()),
            "algo": self.algo.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            "face_count": face_count,
            "frame_count": frame_count,
            "total_frames": self.stats.lock().unwrap_or_else(|e| e.into_inner()).total_frames,
            "original_key": self.original_media_key.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            "cover_key": cover_key,
            "error": self.error.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            "archived": *self.archived.lock().unwrap_or_else(|e| e.into_inner()),
            "original_input": self.original_input.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        })
    }
}

/// 从已上锁的 frames guard(或其切片)统计 face 总数,避免二次加锁。
fn face_count_locked(frames: &[FrameResult]) -> usize {
    frames.iter().map(|f| f.faces.len()).sum()
}

pub struct JobRegistry {
    pub jobs: Mutex<HashMap<String, Arc<Job>>>,
    pub s3: Arc<S3Client>,
    pub cfg: Config,
    pub db: Arc<Db>,
    /// tokio runtime handle,用于在 std::thread 派生的后台任务里
    /// 调用 tokio::spawn 写入 DB(SSE 推送、PG 持久化等)。
    pub rt: tokio::runtime::Handle,
    /// 并发上限。`acquire_owned` 在 spawn_run 之前调用,permit 在 job 结束
    /// 时自动 drop → 自动释放槽位。permit 数 = `cfg.max_concurrent_jobs`。
    pub job_slots: Arc<Semaphore>,
    /// 运行中(已拿到 permit)任务数,配额扣减用;与 semaphore permits 对应。
    pub running_jobs: AtomicU64,
    /// 排队中(已创建未拿到 permit)任务数,背压用。
    /// `max_queue_depth > 0` 时超过即拒绝新任务(HTTP 429)。
    pub queued_jobs: AtomicU64,
    /// 平台进程启动时刻,用于 `/api/metrics` 上报 uptime。
    pub started_at: std::time::Instant,
    /// 停机信号:true 后 spawn_run 不再接收新任务,运行中的收到 cancel。
    pub shutdown: Arc<AtomicBool>,
    /// Persistent gallery state (DB-backed persons + faces). Powers
    /// /api/persons and /api/identify. Hydrated from PG on startup;
    /// writes go straight to PG (the in-memory cache is rebuilt lazily).
    pub gallery: Arc<crate::gallery::GalleryState>,
    /// 增量聚合计数器:由 `metrics` / `job_stats` 端点直接读 O(1),
    /// 不再每次扫描 registry 里全部 job。状态计数在 create / set_status
    /// 同步路径维护,帧计数在 run_job 的 stats 更新分支维护
    /// (per-job stats lock 已经在那里拿过,顺带 bump 原子计数零额外锁开销)。
    /// 读侧 0 次 mutex + 0 次 per-job lock = 缓存未命中也是 O(1)。
    pub agg: Arc<MetricsAgg>,
    /// Per-algo 聚合计数器,按算法名分桶。algo 决定前(`run_job` 早期)
    /// 暂记入 "pending",`build_detector` 成功后迁移到正确 algo 名。
    /// 锁粒度:lazy insert 时短暂持锁,atomic 读 / 写都走 Arc clone。
    pub algo_aggs: Mutex<std::collections::BTreeMap<String, Arc<AlgoAggAtomics>>>,
}

/// 平台级 KPI 聚合的原子容器。`/api/metrics` / `/api/jobs/stats` /
/// `PlatformMetrics::current()` 全都从这里读 O(1),不再扫 registry。
///
/// 一致性:
/// - 状态计数(create / mark_started / set_status)用 SeqCst 顺序,
///   状态机的合法转移是线性的(queued → running → terminal),
///   计数器错一格就把前端 KPI 弄脏;
/// - 帧计数在 run_job 主循环维护,各 job 独立的累加路径,Relaxed
///   就够了 —— 单调递增、无回退语义,聚合时短暂偏差可接受。
#[derive(Debug, Default)]
pub struct MetricsAgg {
    pub total_jobs: AtomicU64,
    pub running: AtomicU64,
    pub queued: AtomicU64,
    pub done: AtomicU64,
    pub errored: AtomicU64,
    pub cancelled: AtomicU64,
    pub total_frames_processed: AtomicU64,
    pub total_frames_with_face: AtomicU64,
    pub total_detections: AtomicU64,
    pub total_spoof_detections: AtomicU64,
    pub total_blocked_detections: AtomicU64,
    /// 终态 job 的 elapsed_ms 累加,`avg_job_ms = sum / timed_count`。
    pub elapsed_sum_ms: AtomicU64,
    pub timed_count: AtomicU64,
}

/// Per-algo 原子聚合,与 `algo_aggs` BTreeMap 配合使用。
/// `Arc` 包裹是为了 lazy insert 后即便 BTreeMap 重新 hash,atomic
/// 地址仍然稳定 —— 计数在 run_job 主循环里增量写,地址变一次就丢更新。
///
/// `active` 不单独存:`active = total - done - cancelled - error`,
/// 读侧算法表达。增量路径里少一个 atomic,迁移逻辑也少一处。
#[derive(Debug, Default)]
pub struct AlgoAggAtomics {
    pub total: AtomicU64,
    pub done: AtomicU64,
    pub cancelled: AtomicU64,
    pub error: AtomicU64,
    pub elapsed_sum_ms: AtomicU64,
    pub timed_count: AtomicU64,
    pub detections: AtomicU64,
}

/// 终态判断:done/cancelled/error 视为已结束,queued/running 仍在跑。
pub fn is_terminal(status: JobStatus) -> bool {
    matches!(
        status,
        JobStatus::Done | JobStatus::Cancelled | JobStatus::Error
    )
}

/// 队列背压:排队深度达到 `MAX_QUEUE_DEPTH` 时 create 返回该错误。
#[derive(Debug)]
pub struct QueueFull {
    pub depth: usize,
    pub max: usize,
}

impl std::fmt::Display for QueueFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "job queue full: {} queued (max {})",
            self.depth, self.max
        )
    }
}

static JOB_COUNTER: AtomicU64 = AtomicU64::new(0);
/// 暂存上传文件名的进程内序号(与时间戳组合,避免重名)。
static UPLOAD_SEQ: AtomicU64 = AtomicU64::new(0);

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn new_id() -> String {
    let c = JOB_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:x}-{:04x}", now_ms(), c)
}

impl JobRegistry {
    /// 创建 job 并登记。背压:`max_queue_depth > 0` 且当前排队数已达上限时
    /// 返回 Err(调用方应回 429)。排队数 = 已创建但还没拿到 semaphore permit
    /// 的任务(不含 running)。
    ///
    /// `self: &Arc<Self>` 是为了让 Job 持 `Weak<JobRegistry>`,set_status
    /// 把状态变化推到聚合计数时不需要走回 caller 又拿一次 Arc。所有调用点
    /// 都是 `Arc<JobRegistry>::create(...)` 直接调用,Rust 自动 borrow。
    pub fn create(self: &Arc<Self>, kind: JobKind, display_name: String) -> Result<Arc<Job>, QueueFull> {
        let max_q = self.cfg.max_queue_depth;
        if max_q > 0 {
            // fetch_add 先占坑再判断:并发下最多短暂超限 1,可接受。
            let now_q = self.queued_jobs.fetch_add(1, Ordering::SeqCst) + 1;
            if now_q > max_q as u64 {
                self.queued_jobs.fetch_sub(1, Ordering::SeqCst);
                return Err(QueueFull {
                    depth: now_q as usize,
                    max: max_q,
                });
            }
        } else {
            // 不限排队深度时也计数,供 /api/metrics 观察。
            self.queued_jobs.fetch_add(1, Ordering::SeqCst);
        }
        // O(1) KPI:total_jobs + queued + 算法桶 pre-seed "pending"。
        // 全部走 atomic,单条 create 路径不引入新锁。
        self.agg.total_jobs.fetch_add(1, Ordering::SeqCst);
        self.agg.queued.fetch_add(1, Ordering::SeqCst);
        self.algo_aggs_inc("pending", |a| {
            a.total.fetch_add(1, Ordering::SeqCst);
        });
        let (tx, _rx) = broadcast::channel(256);
        let dn_for_db = display_name.clone();
        let registry_weak: std::sync::Weak<JobRegistry> = Arc::downgrade(self);
        let job = Arc::new(Job {
            id: new_id(),
            kind,
            display_name,
            created_ms: now_ms(),
            status: Mutex::new(JobStatus::Queued),
            frames: Mutex::new(Vec::new()),
            stats: Mutex::new(JobStats::default()),
            algo: Mutex::new(None),
            algo_override: Mutex::new(None),
            original_media_key: Mutex::new(None),
            error: Mutex::new(None),
            archived: Mutex::new(false),
            original_input: Mutex::new(None),
            cancel: Arc::new(AtomicBool::new(false)),
            event_tx: tx,
            worker: Mutex::new(None),
            recognize_cache: Mutex::new(None),
            registry: registry_weak,
        });
        self.jobs
            .lock()
            .unwrap()
            .insert(job.id.clone(), job.clone());
        let db = self.db.clone();
        let id = job.id.clone();
        let dn = dn_for_db;
        let kind_db = job.kind;
        let created_ms = job.created_ms;
        tokio::spawn(async move {
            db.insert_job(&id, kind_db, &dn, JobStatus::Queued, created_ms)
                .await;
        });
        Ok(job)
    }

    /// 任务从 queued → running 时调用(permit 已拿到)。
    fn mark_started(&self) {
        self.queued_jobs.fetch_sub(1, Ordering::SeqCst);
        self.running_jobs.fetch_add(1, Ordering::SeqCst);
        self.agg.queued.fetch_sub(1, Ordering::SeqCst);
        self.agg.running.fetch_add(1, Ordering::SeqCst);
        // algo_aggs 不动:queued→running 不涉及桶迁移
        // (algo 在 run_job 早期 build_detector 后才确定,那时
        // 不论 job 当前是 queued 还是 running,都一次性把桶从 pending
        // 迁到具体 algo —— `transition_algo_bucket` 处理)。
    }

    /// 任务线程退出(任何终态)时调用。
    fn mark_finished(&self) {
        self.running_jobs.fetch_sub(1, Ordering::SeqCst);
        self.agg.running.fetch_sub(1, Ordering::SeqCst);
    }

    /// 排队阶段即被取消:不进入 running,只回退 queued 计数。
    fn mark_abandoned(&self) {
        self.queued_jobs.fetch_sub(1, Ordering::SeqCst);
        self.agg.queued.fetch_sub(1, Ordering::SeqCst);
        // pending.total 也要回退 1,因为本 job 永远不会进入 run_job 走迁移。
        // 注意:只在 mark_abandoned 路径减 pending.total;transition_algo_bucket
        // 不动(没新桶接收)。让 pending.total 始终准确。
        self.algo_aggs_inc("pending", |a| {
            a.total.fetch_sub(1, Ordering::SeqCst);
        });
    }

    /// Job 状态转移时(由 `Job::set_status` 调)同步更新聚合计数。
    /// `prev == s` 时调用方已 short-circuit,这里不再判等。
    ///
    /// 只关心 terminal 转移(running → done/cancelled/error),因为 queued
    /// 和 running 的 active 计数已经在 mark_started / mark_abandoned 维护。
    fn transition_status_counter(&self, prev: JobStatus, s: JobStatus) {
        // 仅在 prev=Running → s=Terminal 这条路径上有意义。
        if prev == JobStatus::Running && is_terminal(s) {
            match s {
                JobStatus::Done => {
                    self.agg.done.fetch_add(1, Ordering::SeqCst);
                }
                JobStatus::Cancelled => {
                    self.agg.cancelled.fetch_add(1, Ordering::SeqCst);
                }
                JobStatus::Error => {
                    self.agg.errored.fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
        }
    }

    /// 帧级增量:在 run_job 主循环的 stats 更新分支同步 bump。
    /// 调用方已经持 per-job stats lock,bump 一次 atomic 不再额外加锁。
    pub fn bump_frame_counters(&self, frames_inc: u64, faces_inc: u64, dets_inc: u64) {
        if frames_inc > 0 {
            self.agg
                .total_frames_processed
                .fetch_add(frames_inc, Ordering::Relaxed);
        }
        if faces_inc > 0 {
            self.agg
                .total_frames_with_face
                .fetch_add(faces_inc, Ordering::Relaxed);
        }
        if dets_inc > 0 {
            self.agg
                .total_detections
                .fetch_add(dets_inc, Ordering::Relaxed);
        }
    }

    /// 活体 / 拦截统计增量:spoof 和 blocked 在 liveness 路径上 bump。
    pub fn bump_liveness_counters(&self, spoof_inc: u64, blocked_inc: u64) {
        if spoof_inc > 0 {
            self.agg
                .total_spoof_detections
                .fetch_add(spoof_inc, Ordering::Relaxed);
        }
        if blocked_inc > 0 {
            self.agg
                .total_blocked_detections
                .fetch_add(blocked_inc, Ordering::Relaxed);
        }
    }

    /// Job 进入终态时:把 elapsed_ms 累加进 avg 计数器。
    /// `finalize` 之后调一次(写 stats.elapsed_ms 的同帧)。
    pub fn record_terminal_elapsed(&self, elapsed_ms: u64) {
        if elapsed_ms > 0 {
            self.agg.elapsed_sum_ms.fetch_add(elapsed_ms, Ordering::Relaxed);
            self.agg.timed_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Per-algo 桶的 helper:`algo_aggs` 锁内 lazy insert + 持锁外调用
    /// 用户给定的 closure 对 atomics 做修改。读侧 `algo_aggs_snapshot` 直接
    /// 复制 BTreeMap 内容 + clone Arc。
    fn algo_aggs_inc(&self, name: &str, f: impl FnOnce(&AlgoAggAtomics)) {
        let mut map = self.algo_aggs.lock().unwrap_or_else(|e| e.into_inner());
        let entry = map
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(AlgoAggAtomics::default()))
            .clone();
        drop(map);
        f(&entry);
    }

    /// 把 job 从 `pending` 桶迁到具体算法桶。`run_job` `build_detector`
    /// 成功后调一次。active 不单独存(读侧算法表达 total - 终态),所以
    /// 这里只搬 `total`:pending 减 1,新 algo 加 1。
    ///
    /// 重复调用(同 job)幂等保护:已迁移过的 pending.total 减过头。`old`
    /// 是 `Job::algo` 当前值(可能就是 "pending",也可能已经是某 algo 名,
    /// 取决于调用方)。重复调用 old == new 时 short-circuit。
    pub fn transition_algo_bucket(&self, old: &str, new: &str) {
        if old == new {
            return;
        }
        // 旧桶减 1,新桶加 1。两者都可能 lazy 创建。
        self.algo_aggs_inc(old, |a| {
            a.total.fetch_sub(1, Ordering::SeqCst);
        });
        self.algo_aggs_inc(new, |a| {
            a.total.fetch_add(1, Ordering::SeqCst);
        });
    }

    /// 给 job 的具体 algo 桶加 detection 计数(`run_job` 主循环里同步 bump)。
    /// 老 path 走 `aggregate_algo_stats` 一次性扫 + 加,这里换成增量。
    pub fn bump_algo_detections(&self, algo: &str, dets_inc: u64) {
        if dets_inc == 0 {
            return;
        }
        self.algo_aggs_inc(algo, |a| {
            a.detections.fetch_add(dets_inc, Ordering::Relaxed);
        });
    }

    /// 终态时把 elapsed_ms 推到具体 algo 桶(供 per-algo avg)。
    pub fn record_algo_elapsed(&self, algo: &str, elapsed_ms: u64) {
        if elapsed_ms == 0 {
            return;
        }
        self.algo_aggs_inc(algo, |a| {
            a.elapsed_sum_ms.fetch_add(elapsed_ms, Ordering::Relaxed);
            a.timed_count.fetch_add(1, Ordering::Relaxed);
        });
    }

    /// 终态 done / cancelled / error 时,把对应 algo 桶的状态转移走完。
    /// active 已经在 transition_algo_bucket + mark_finished 路径里减过,
    /// 这里只 bump done/cancelled/error 桶。
    pub fn finalize_algo_status(&self, algo: &str, terminal: JobStatus) {
        match terminal {
            JobStatus::Done => self.algo_aggs_inc(algo, |a| {
                a.done.fetch_add(1, Ordering::SeqCst);
            }),
            JobStatus::Cancelled => self.algo_aggs_inc(algo, |a| {
                a.cancelled.fetch_add(1, Ordering::SeqCst);
            }),
            JobStatus::Error => self.algo_aggs_inc(algo, |a| {
                a.error.fetch_add(1, Ordering::SeqCst);
            }),
            _ => {}
        }
    }

    /// 读侧:O(1) 快照,把 BTreeMap 复制成一对 (name, snapshot) 的 Vec。
    /// 每个 snapshot 是 `AlgoAgg`(从原子读回,见下)。
    pub fn algo_aggs_snapshot(&self) -> Vec<(String, AlgoAgg)> {
        let map = self.algo_aggs.lock().unwrap_or_else(|e| e.into_inner());
        map.iter()
            .map(|(name, atomics)| {
                let a = atomics;
                let total = a.total.load(Ordering::Relaxed);
                let done = a.done.load(Ordering::Relaxed);
                let cancelled = a.cancelled.load(Ordering::Relaxed);
                let error = a.error.load(Ordering::Relaxed);
                let active = total.saturating_sub(done + cancelled + error);
                (
                    name.clone(),
                    AlgoAgg {
                        total,
                        done,
                        cancelled,
                        error,
                        active,
                        elapsed_ms_sum: a.elapsed_sum_ms.load(Ordering::Relaxed),
                        timed_count: a.timed_count.load(Ordering::Relaxed),
                        detections: a.detections.load(Ordering::Relaxed),
                    },
                )
            })
            .collect()
    }

    /// O(1) 平台级 KPI 快照:`/api/metrics` 直接返回这个,无 registry 扫描。
    pub fn metrics_snapshot(&self) -> PlatformMetrics {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let started_ms = now_ms.saturating_sub(self.started_at.elapsed().as_millis() as u64);
        let a = &self.agg;
        let total_jobs = a.total_jobs.load(Ordering::SeqCst);
        let running = a.running.load(Ordering::SeqCst);
        let queued = a.queued.load(Ordering::SeqCst);
        let done = a.done.load(Ordering::SeqCst);
        let errored = a.errored.load(Ordering::SeqCst);
        let cancelled = a.cancelled.load(Ordering::SeqCst);
        let total_frames_processed = a.total_frames_processed.load(Ordering::Relaxed);
        let total_frames_with_face = a.total_frames_with_face.load(Ordering::Relaxed);
        let total_detections = a.total_detections.load(Ordering::Relaxed);
        let total_spoof_detections = a.total_spoof_detections.load(Ordering::Relaxed);
        let total_blocked_detections = a.total_blocked_detections.load(Ordering::Relaxed);
        let elapsed_sum_ms = a.elapsed_sum_ms.load(Ordering::Relaxed);
        let timed_count = a.timed_count.load(Ordering::Relaxed);
        // 高级字段仍未在 `MetricsAgg` 上建模(预留),保持 0 兼容前端契约。
        let total_gpu_levels = 0u64;
        let total_cpu_levels = 0u64;
        let total_gpu_skipped_levels = 0u64;
        let total_levels = total_gpu_levels + total_cpu_levels;
        let gpu_pct = if total_levels > 0 {
            (total_gpu_levels as f32) * 100.0 / (total_levels as f32)
        } else {
            0.0
        };
        let live_fps_max = 0.0f32;
        let live_fps_avg = 0.0f32;
        let total_cascade_evals = 0u64;
        let total_cascade_passes = 0u64;
        let cascade_pass_rate = if total_cascade_evals > 0 {
            total_cascade_passes as f32 / total_cascade_evals as f32
        } else {
            0.0
        };
        let avg_job_ms = if timed_count > 0 {
            elapsed_sum_ms as f64 / timed_count as f64
        } else {
            0.0
        };
        let max_conc = self.cfg.max_concurrent_jobs as u32;
        let concurrency = running.min(max_conc as u64) as u32;
        let mode = std::env::var("RSFACE_ALGO")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                if self.cfg.use_cnn || self.cfg.cnn_weights.is_some() {
                    "cnn".into()
                } else {
                    "haar".into()
                }
            });
        PlatformMetrics {
            started_ms,
            now_ms,
            total_jobs,
            running,
            queued,
            done,
            errored,
            cancelled,
            total_frames_processed,
            total_frames_with_face,
            total_detections,
            total_spoof_detections,
            total_blocked_detections,
            total_gpu_levels,
            total_cpu_levels,
            total_gpu_skipped_levels,
            gpu_pct,
            live_fps_max,
            live_fps_avg,
            total_cascade_evals,
            total_cascade_passes,
            cascade_pass_rate,
            avg_job_ms,
            concurrency,
            max_concurrency: max_conc,
            db_enabled: self.db.pool.is_some(),
            s3_endpoint: self.cfg.s3_endpoint.clone(),
            mode,
        }
    }
    /// 删除任务媒体:fire-and-forget 地在 blocking 线程枚举并删除
    /// S3 对象 + 本地 media/tmp 目录。任务已先从索引移除,清理结果
    /// 不影响删除响应;个别对象删除失败只在服务端日志告警。
    pub fn spawn_media_cleanup(&self, id: &str) {
        let s3 = self.s3.clone();
        let cfg = self.cfg.clone();
        let jid = id.to_string();
        // 中 #5:接管 worker JoinHandle,join 完再删 S3(防止 run_job 末尾还在写)。
        let worker = self
            .jobs
            .lock()
            .unwrap()
            .get(id)
            .and_then(|j| j.worker.lock().unwrap_or_else(|p| p.into_inner()).take());
        tokio::task::spawn_blocking(move || {
            cleanup_job_media_blocking(&s3, &cfg, &jid, worker);
        });
    }

    /// 优雅停机:拒绝新任务 + 对所有 queued/running 任务发 cancel。
    pub fn begin_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let jobs = self.list();
        let n = jobs.len();
        for j in jobs {
            if !is_terminal(j.status()) {
                j.cancel.store(true, Ordering::SeqCst);
            }
        }
        tracing::info!("[jobs] shutdown: cancel signalled to non-terminal jobs (total {n})");
    }

    /// 是否还有 queued/running 任务(停机排空检测)。
    pub fn active_count(&self) -> usize {
        self.jobs
            .lock()
            .unwrap()
            .values()
            .filter(|j| !is_terminal(j.status()))
            .count()
    }

    /// 设置 job 的原始输入路径(URL / 文件名),供"重试"使用。
    pub fn set_original_input(&self, id: &str, input: String) {
        if let Some(j) = self.jobs.lock().unwrap().get(id).cloned() {
            *j.original_input.lock().unwrap_or_else(|e| e.into_inner()) = Some(input);
        }
    }

    /// 设置 job 的算法覆盖(来自 upload 表单的 `algo` 字段)。
    /// 仅在名字属于 `available_algos()` 时生效;否则忽略(回退到 env/默认)。
    pub fn set_algo_override(&self, id: &str, algo: String) {
        if let Some(j) = self.jobs.lock().unwrap().get(id).cloned() {
            let lower = algo.trim().to_ascii_lowercase();
            if available_algos().contains(&lower.as_str()) {
                *j.algo_override.lock().unwrap_or_else(|e| e.into_inner()) = Some(lower);
            }
        }
    }

    /// 标记 job 为已归档(侧栏默认隐藏但仍在内存/DB 中)。
    /// 内存标志翻转 + DB 列同步落库,确保重启后归档状态一致。
    pub fn set_archived(&self, id: &str, archived: bool) -> bool {
        if let Some(j) = self.jobs.lock().unwrap().get(id).cloned() {
            *j.archived.lock().unwrap_or_else(|e| e.into_inner()) = archived;
            // Persist the archive flag flip so the row matches the in-memory
            // state across restarts (jobs health columns migration).
            let db = self.db.clone();
            let id_db = id.to_string();
            tokio::spawn(async move {
                db.set_archived(&id_db, archived).await;
            });
            true
        } else {
            false
        }
    }

    pub fn get(&self, id: &str) -> Option<Arc<Job>> {
        self.jobs.lock().unwrap().get(id).cloned()
    }

    pub fn list(&self) -> Vec<Arc<Job>> {
        let mut v: Vec<Arc<Job>> = self.jobs.lock().unwrap().values().cloned().collect();
        v.sort_by_key(|j| std::cmp::Reverse(j.created_ms));
        v
    }

    /// 取消正在跑的 job(写 cancel flag),让 watchdog 触发 Cancelled 状态。
    /// 用于 archive 前先停下任务。
    pub fn request_cancel(&self, id: &str) -> bool {
        if let Some(j) = self.jobs.lock().unwrap().get(id).cloned() {
            j.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// 从内存索引中删除 job。DB 删除由调用方负责。
    /// 返回是否实际删除(true=存在并删除,false=不存在)。
    pub fn remove(&self, id: &str) -> bool {
        self.jobs.lock().unwrap().remove(id).is_some()
    }

    /// 批量删除(ids → `Vec<bool>` 表示每个 id 是否被删)。
    pub fn remove_many(&self, ids: &[String]) -> Vec<bool> {
        let mut jobs = self.jobs.lock().unwrap();
        ids.iter().map(|id| jobs.remove(id).is_some()).collect()
    }

    /// 排队阶段失败时调用:从索引删除**并**回退 `queued_jobs` 计数。
    ///
    /// 只有当任务确实还在索引里(即尚未 `mark_started` 进入 running)才回退计数,
    /// 因此重复调用、或对已进入 running 的任务调用都不会让计数器失衡。
    /// 返回任务是否存在;DB 行的删除由调用方负责。
    pub fn abandon_job(&self, id: &str) -> bool {
        if self.remove(id) {
            self.mark_abandoned();
            true
        } else {
            false
        }
    }

    /// 启动后台检测线程。`input` 为本地文件路径(上传落盘)或 URL(直播流)。
    ///
    /// **并发控制**:在 std::thread::spawn 之前先通过 `job_slots.acquire_owned().await`
    /// 拿一个 permit;permit 在函数返回时自动 drop 释放。permit 拿不到时,**调用方
    /// 仍会立刻返回**(job 状态进入 Queued),permit 一旦就绪,后台线程自动启动 —
    /// 实现"满 N 时排队"的语义,而不是阻塞 HTTP 路径。
    ///
    /// **Panic 隔离**:整个 run_job 用 `std::panic::catch_unwind` 包住,panic 后
    /// 状态转 `JobStatus::Error`,SSE 发 error 事件,server 不会跟着挂。
    ///
    /// **超时 watchdog**:根据 job 类型计算 timeout,后台另起一线程轮询
    /// `cancel` 标志;超时置位 cancel,run_job 内部检测到后转 Cancelled。
    pub fn spawn_run(self: &Arc<Self>, job: Arc<Job>, input: String) {
        let reg = self.clone();
        let rt = reg.rt.clone();
        let semaphore = self.job_slots.clone();
        let timeout_secs = match job.kind {
            // 流任务默认不限(由 cancel 控),但 JOB_TIMEOUT_STREAM_SECS>0
            // 时可给挂死流加硬超时(可配置)。
            JobKind::Stream => reg.cfg.job_timeout_stream_secs,
            JobKind::Video => reg.cfg.job_timeout_video_secs,
            JobKind::Image => reg.cfg.job_timeout_secs,
        };
        // 异步等 permit;permit 拿到后真正启动检测线程。
        // cancel 始终由调用方控制,即使没拿到 permit,仍能 cancel 一个排队 job。
        tokio::spawn(async move {
            // 排队期间被 cancel / 停机:直接转 Cancelled,不占用检测线程。
            if job.cancel.load(Ordering::SeqCst) {
                job.set_status(JobStatus::Cancelled);
                job.emit(&serde_json::json!({"type": "cancelled"}).to_string());
                reg.mark_abandoned();
                // 低 #14:同 terminated 路径一样写 DB,避免重启后行残留 Queued。
                let db = reg.db.clone();
                let id = job.id.clone();
                tokio::spawn(async move {
                    db.update_job_status(
                        &id,
                        JobStatus::Cancelled,
                        Some(crate::persist::now_ms_u64()),
                        None,
                    )
                    .await;
                });
                return;
            }
            let permit = match semaphore.acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    // semaphore 被 close,理论上 server 关闭时才会发生。
                    let _ = job;
                    return;
                }
            };
            // permit 拿到后再查一次 cancel:排队期间用户可能已点取消。
            if job.cancel.load(Ordering::SeqCst) {
                job.set_status(JobStatus::Cancelled);
                job.emit(&serde_json::json!({"type": "cancelled"}).to_string());
                reg.mark_abandoned(); // permit drop 时释放槽位
                return;
            }
            reg.mark_started();
            // 把 permit 搬到 std::thread 里,thread 结束(成功/panic/超时)时
            // permit 自动 drop → 释放槽位。
            let job_for_handle = job.clone();
            let handle = std::thread::spawn(move || {
                let _permit = permit; // 关键:绑在 stack 上,thread 退出 → drop
                let _guard = rt.enter();
                // 线程任何退出路径都先扣 running 计数(permit 的 drop 在此之后)。
                struct FinishGuard(Arc<JobRegistry>);
                impl Drop for FinishGuard {
                    fn drop(&mut self) {
                        self.0.mark_finished();
                    }
                }
                let _finish = FinishGuard(reg.clone());
                // ----- 超时 watchdog -----
                let cancel_handle = if timeout_secs > 0 {
                    let cancel = job.cancel.clone();
                    let id = job.id.clone();
                    let secs = timeout_secs;
                    let h = std::thread::spawn(move || {
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(secs);
                        loop {
                            let now = std::time::Instant::now();
                            if now >= deadline {
                                tracing::warn!(
                                    "[job {id}] watchdog: timeout after {secs}s, cancelling"
                                );
                                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                                break;
                            }
                            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                                break; // 任务已主动结束
                            }
                            let left = deadline.saturating_duration_since(now);
                            // 100ms~5s 之间的轮询间隔,避免 sleep 太长错过 cancel。
                            let wait = left.min(std::time::Duration::from_millis(500));
                            std::thread::sleep(wait);
                        }
                    });
                    Some(h)
                } else {
                    None
                };
                // ----- 主任务:panic 隔离 -----
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    reg.run_job(&job, &input)
                }));
                // run_job 已结束(成功 / 错误 / 主动 cancel)。立刻把 cancel 置位,
                // 让 watchdog 看到 true 后立即 break,不再等 timeout 走完。
                // 副作用:job 状态机进入终态后 cancel 标志也是 true,但 cancel API
                // 的语义本来就是"取消信号已记录",不影响行为。
                job.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                // 限时 join watchdog:1s 内没醒就 detach,绝不让它阻塞 permit 释放。
                // 修复 semaphore 卡死:之前 `h.join()` 会等 watchdog 走完整个
                // timeout(120s / 300s),permit 一直持有,后续 job 全卡 queued。
                if let Some(h) = cancel_handle {
                    let join_deadline =
                        std::time::Instant::now() + std::time::Duration::from_secs(1);
                    loop {
                        if h.is_finished() {
                            let _ = h.join();
                            break;
                        }
                        if std::time::Instant::now() >= join_deadline {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                }
                match result {
                    Ok(Ok(())) => { /* run_job 自己处理 success */ }
                    Ok(Err(e)) => {
                        *job.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(e.to_string());
                        job.set_status(JobStatus::Error);
                        job.emit(
                            &serde_json::json!({"type": "error", "message": e.to_string()})
                                .to_string(),
                        );
                        tracing::warn!("[job {}] error: {e}", job.id);
                        let err_str = e.to_string();
                        let id = job.id.clone();
                        let db = reg.db.clone();
                        tokio::spawn(async move {
                            db.update_job_status(
                                &id,
                                JobStatus::Error,
                                Some(crate::persist::now_ms_u64()),
                                Some(&err_str),
                            )
                            .await;
                        });
                    }
                    Err(panic_info) => {
                        // panic 转 Error 状态,避免拖垮 server。
                        let msg = panic_message(&panic_info);
                        *job.error.lock().unwrap_or_else(|e| e.into_inner()) =
                            Some(format!("panic: {msg}"));
                        job.set_status(JobStatus::Error);
                        job.emit(&serde_json::json!({"type": "error", "message": format!("panic: {msg}")}).to_string());
                        tracing::warn!("[job {}] PANIC: {msg}", job.id);
                        let err_str = format!("panic: {msg}");
                        let id = job.id.clone();
                        let db = reg.db.clone();
                        tokio::spawn(async move {
                            db.update_job_status(
                                &id,
                                JobStatus::Error,
                                Some(crate::persist::now_ms_u64()),
                                Some(&err_str),
                            )
                            .await;
                        });
                    }
                }
            });
            // 中 #5:把 JoinHandle 存进 job.worker;delete 时 join 一下,确保
            // 后台线程不会再写新的 S3 对象,然后再清理 prefix。
            *job_for_handle
                .worker
                .lock()
                .unwrap_or_else(|p| p.into_inner()) = Some(handle);
        });
    }

    fn run_job(&self, job: &Arc<Job>, input: &str) -> std::io::Result<()> {
        let started = std::time::Instant::now();
        job.set_status(JobStatus::Running);
        job.emit(&serde_json::json!({"type": "status", "status": "running"}).to_string());
        {
            let db = self.db.clone();
            let id = job.id.clone();
            tokio::spawn(async move {
                // mark_started 同时写 status='running' + started_at +
                // heartbeat_at。heartbeat 是后续 orphan 回收的依据。
                db.mark_started(&id, crate::persist::now_ms_u64()).await;
            });
        }

        // SSE 心跳(per-job):每秒用 frames_processed 增量算 fps,推给前端。
        // 纯 SSE emit,**不写 DB** —— DB 心跳统一交给 `spawn_heartbeat_coalescer`
        // 每 5s 一次批量 UPDATE(N 个 running job 共 1 次 roundtrip,而
        // 不是每 job 各 2 次)。SSE 推送仍需要 per-job stats 锁拿
        // `frames_processed`,跨 job 不好算 fps,所以这条路径不动。
        {
            let cancel = job.cancel.clone();
            let event_tx = job.event_tx.clone();
            let job_for_sse = job.clone();
            tokio::spawn(async move {
                // 第一次 tick 立刻发一个 0fps 心跳:让前端的 fps 卡片立刻
                // 出现,不会等到 1s 后才第一次更新。
                if !cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = event_tx.send(
                        serde_json::json!({
                            "type": "heartbeat",
                            "fps": 0.0_f32,
                            "frames_processed": job_for_sse
                                .stats
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .frames_processed,
                        })
                        .to_string(),
                    );
                }
                let mut tick_sse = tokio::time::interval(std::time::Duration::from_secs(1));
                tick_sse.tick().await; // 跳过首次立即触发(上面手动发过)
                let mut last_frames: u64 = job_for_sse
                    .stats
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .frames_processed;
                let mut last_ts = Instant::now();
                loop {
                    tick_sse.tick().await;
                    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let now = Instant::now();
                    let frames_now = job_for_sse
                        .stats
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .frames_processed;
                    let dt_ms = now.saturating_duration_since(last_ts).as_millis() as u64;
                    let df = frames_now.saturating_sub(last_frames);
                    // dt_ms = 0 时 fps=0(系统时钟回退或同 ms 内连发);
                    // 显式 clamp 到 1000 fps 防异常广播峰值。
                    let fps = if dt_ms > 0 {
                        let f = (df as f32) * 1000.0 / (dt_ms as f32);
                        f.min(1000.0)
                    } else {
                        0.0
                    };
                    last_frames = frames_now;
                    last_ts = now;
                    let _ = event_tx.send(
                        serde_json::json!({
                            "type": "heartbeat",
                            "fps": fps,
                            "frames_processed": frames_now,
                        })
                        .to_string(),
                    );
                }
            });
        }

        // 1) 输入预处理:图片归一化为 PGM;视频 URL 同步 ffmpeg 转 mp4;
        //    流 URL 后台 ffmpeg 持续写 mp4,core 反复读增长文件。
        let work_dir = self.cfg.tmp_dir.join(&job.id);
        std::fs::create_dir_all(&work_dir)?;
        let mut stream_proc: Option<Child> = None;
        let input_path = match job.kind {
            JobKind::Image => normalize_image_input(input, &work_dir)?,
            JobKind::Video => {
                if is_core_native_video(input) {
                    input.to_string()
                } else {
                    // 视频 URL:同步 ffmpeg 跑完后再喂给 core。等待过程可被
                    // 用户取消/停机打断(旧代码 child.wait() 不可中断,长转码会
                    // 一直跑到结束);取消时 kill 子进程并让本任务提前失败。
                    let (path, child) = spawn_ffmpeg_to_local(input, &work_dir, "video.mp4")?;
                    wait_child_cancellable(child, &job.cancel)?;
                    path.to_string_lossy().to_string()
                }
            }
            JobKind::Stream => {
                // 流:ffmpeg 后台持续写,core 反复打开读取增长文件。
                let (path, child) = spawn_ffmpeg_to_local(input, &work_dir, "stream.mp4")?;
                stream_proc = Some(child);
                path.to_string_lossy().to_string()
            }
        };
        let _keep_stream = StreamGuard(stream_proc); // 任务结束自动 kill 进程

        // 3) 原始媒体写入存储(S3 失败自动降级本地)。
        //    如果 handle_upload 阶段已经预存了用户的原始文件(用于
        //    PNG/JPG 转 PGM 后 web preview),这里就跳过。
        if job.kind != JobKind::Stream
            && job
                .original_media_key
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_none()
        {
            let ext = Path::new(&input_path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("bin")
                .to_ascii_lowercase();
            let key = format!("jobs/{}/original.{ext}", job.id);
            let ct = match ext.as_str() {
                "png" => "image/png",
                "mp4" => "video/mp4",
                "webm" => "video/webm",
                "mov" => "video/quicktime",
                "mkv" => "video/x-matroska",
                _ => "application/octet-stream",
            };
            // 高 #2:流式 PUT —— `fs::read` + `bytes.to_vec` 在大视频上传时
            // 是 ~2× 文件大小的堆峰值;改走 `put_object_file` (UNSIGNED-PAYLOAD
            // 签名),GB 级文件不再进内存。
            let stored = put_with_fallback_file(self, &key, ct, Path::new(&input_path));
            *job.original_media_key
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(stored.clone());
            {
                let db = self.db.clone();
                let id = job.id.clone();
                tokio::spawn(async move {
                    db.set_original_key(&id, &stored).await;
                });
            }
        }

        // 3) 检测器:按 cfg + RSFACE_ALGO 选择 3 种 zero-dep 算法之一(haar/cnn/luminance)。
        //    全部对外暴露同一份 `detect(gray) -> Vec<Detection>` 接口,后续
        //    frame 循环无需分支。SSE 事件同时带 `mode` (旧) 和 `algo` (新)
        //    字段,前端新老代码都能识别。
        //
        // 早期记录 attempted algo:即使 detector 构建失败(典型场景:用户
        // 选 haar 但 cascade.rfcf 缺失),前端 /api/jobs 摘要里也要显示
        // "尝试的是哪个",而不是 null — 否则 UI 没法定位是哪条配置导致
        // 的 failure。
        if let Some(override_name) = job
            .algo_override
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            *job.algo.lock().unwrap_or_else(|e| e.into_inner()) = Some(override_name);
        }
        let detector = match build_detector(
            &self.cfg,
            job.algo_override
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_deref(),
        ) {
            Ok(d) => d,
            Err(e) => {
                // 把 attempted algo 写进 stats,前端 chip 仍能过滤
                if let Some(a) = job.algo.lock().unwrap_or_else(|e| e.into_inner()).clone() {
                    job.stats.lock().unwrap_or_else(|e| e.into_inner()).algo = a;
                }
                return Err(e);
            }
        };
        // 记录实际使用的算法,供 /api/jobs 摘要里的 algo 字段(算法过滤 chip 用)
        let algo_name = detector.kind_name().to_string();
        *job.algo.lock().unwrap_or_else(|e| e.into_inner()) = Some(algo_name.clone());
        {
            let db = self.db.clone();
            let id = job.id.clone();
            let a = algo_name.clone();
            tokio::spawn(async move {
                db.set_algo(&id, &a).await;
            });
        }
        // 同时把算法名记进 stats:供 /api/jobs/stats 按算法聚合 + 前端 chip 过滤。
        {
            let mut st = job.stats.lock().unwrap_or_else(|e| e.into_inner());
            if st.algo.is_empty() {
                st.algo = detector.kind_name().to_string();
            }
        }
        // 算法桶迁移:create() 时此 job 入了 "pending" 桶,现在确定 algo 后
        // 把 total 计数从 pending 搬到具体 algo。重复调用 short-circuit 保护。
        // 注意:本 job 当前 active 还在 running(queued→running 的 mark_started
        // 不动 algo_aggs),total 计数从 pending 搬到 algo 即可,active = total
        // - 终态 在读侧自然正确。
        self.transition_algo_bucket("pending", &algo_name);
        job.emit(
            &serde_json::json!({
                "type": "detector",
                "mode": detector.kind_name(),
                "algo": detector.kind_name(),
                "available_algos": available_algos(),
            })
            .to_string(),
        );

        // 4) 打开源,逐帧检测。流任务在 EOF 时重开(本地 mp4 还在被 ffmpeg 持续写)。
        let mut source = open_source(&input_path)?;
        // 视频总帧数(ffprobe):写入 stats,供前端进度条显示正确分母。
        // 流 / 探测失败为 None,前端降级只显示已处理帧数。
        if let Some(hint) = source.total_hint() {
            let mut st = job.stats.lock().unwrap_or_else(|e| e.into_inner());
            st.total_frames = hint;
        }
        let is_stream = job.kind == JobKind::Stream;
        let max_frames = match job.kind {
            JobKind::Stream => self.cfg.max_frames_stream,
            _ => self.cfg.max_frames_video,
        };
        let mut frame_idx: u64 = 0;
        let mut _frames_since_reopen: u64 = 0;
        // Temporal liveness gate (video/stream only): a face must be judged
        // real on N consecutive frames before enforcement accepts it. Off by
        // default (`liveness_temporal_frames == 1`) and only meaningful with
        // a live liveness backend. Fail-closed: warm-up frames are not
        // confirmed, so a short clip with one good frame cannot pass.
        let temporal_n = self.cfg.liveness_temporal_frames;
        let use_temporal = temporal_n > 1
            && self.gallery.has_liveness()
            && matches!(job.kind, JobKind::Video | JobKind::Stream);
        let mut face_tracker = rsface::tracker::FaceTracker::default();
        // Per-track streak paired with the frame it was last seen, so tracks
        // that leave the frame can be pruned instead of accumulating for the
        // whole (possibly endless) live stream.
        let mut temporal_votes: std::collections::HashMap<
            u32,
            (rsface::liveness::TemporalVote, u64),
        > = std::collections::HashMap::new();
        let temporal_cfg = rsface::liveness::TemporalConfig {
            required_frames: temporal_n,
            min_real_score: self
                .cfg
                .liveness_temporal_min_score
                .unwrap_or(self.cfg.liveness_min_real_score),
        };
        // Drop a streak after this many frames unseen; comfortably above the
        // tracker's `max_age` so a briefly occluded face is not forgotten.
        let temporal_retire_after = 60u64;
        // 批量 DB 写缓冲(20 帧一批;cancel/done 路径都会 flush)。
        let mut db_pending: Vec<FrameResult> = Vec::new();
        // 已产出的人脸裁剪总数(跨帧累计)。旧实现每帧都 lock frames 把
        // 历史所有帧的 faces.len() 重算一遍,长视频上是 O(n²);循环外维护
        // 一个单调计数器即可,与 frames 里的实际裁剪数始终一致。
        let mut total_crops: usize = 0;
        loop {
            if job.cancel.load(Ordering::Relaxed) {
                job.set_status(JobStatus::Cancelled);
                job.emit(&serde_json::json!({"type": "cancelled"}).to_string());
                {
                    let db = self.db.clone();
                    let id = job.id.clone();
                    let st = job.stats.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    tokio::spawn(async move {
                        db.update_job_status(
                            &id,
                            JobStatus::Cancelled,
                            Some(crate::persist::now_ms_u64()),
                            None,
                        )
                        .await;
                        db.update_job_stats(&id, &st).await;
                    });
                }
                // cancel 也要把攒的帧落库。
                if !db_pending.is_empty() {
                    let db = self.db.clone();
                    let jid = job.id.clone();
                    tokio::spawn(async move {
                        db.add_frames_batch(&jid, &db_pending).await;
                    });
                }
                finalize(
                    job,
                    started,
                    &self.cfg.tmp_dir,
                    self,
                    &algo_name,
                    JobStatus::Cancelled,
                );
                return Ok(());
            }
            if max_frames > 0 && frame_idx >= max_frames {
                break;
            }
            let frame = match source.next_frame()? {
                Some(f) => f,
                None => {
                    if is_stream && !job.cancel.load(Ordering::Relaxed) {
                        // 等一下,重开(ffmpeg 还在写)。
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        match open_source(&input_path) {
                            Ok(s2) => {
                                source = s2;
                                _frames_since_reopen = 0;
                                continue;
                            }
                            Err(_) => break,
                        }
                    }
                    break;
                }
            };
            _frames_since_reopen += 1;
            let detections = detector.detect(&frame.gray);
            let has_face = !detections.is_empty();
            let mut frame_spoof = 0u64;
            let mut frame_blocked = 0u64;
            let keepalive = job.kind == JobKind::Stream
                && frame_idx.is_multiple_of(self.cfg.stream_keepalive_period.max(1));

            if has_face || (job.kind == JobKind::Image) || keepalive {
                let base = frame_rgb(&frame);
                let mut result = FrameResult {
                    index: frame.index,
                    timestamp_ms: frame.timestamp_ms,
                    annotated_key: None,
                    original_key: None,
                    faces: Vec::new(),
                };
                // 跟踪 inline:// 数据(优雅降级)。key -> base64 bytes
                let mut inlines: Vec<(String, String)> = Vec::new();

                // 原始帧(直播流 keepalive / 图片对比)。
                if job.kind == JobKind::Stream || job.kind == JobKind::Image {
                    let key = format!("jobs/{}/frames/{:06}.png", job.id, frame.index);
                    if let Some(bytes) = encode_png(&base) {
                        let stored = put_with_inline_fallback(self, &key, "image/png", &bytes);
                        if let Some(b64) = inline_data_b64(&stored, &bytes) {
                            inlines.push((stored.clone(), b64));
                        }
                        result.original_key = Some(stored);
                    }
                }

                // 标注帧。
                // Compute liveness once per detection; annotation colour and the
                // face-crop entries below share these verdicts.
                let tracked = if use_temporal {
                    face_tracker.update(&detections)
                } else {
                    Vec::new()
                };
                let mut verdicts: Vec<(Option<crate::liveness::LivenessVerdict>, bool)> =
                    Vec::with_capacity(detections.len());
                for (idx, d) in detections.iter().enumerate() {
                    let live = self.gallery.check_liveness(&base, d);
                    let enforce = self.gallery.liveness_enforce();
                    let mut blocked = crate::liveness::enforce_blocks(
                        enforce,
                        self.gallery.has_liveness(),
                        live.as_ref().map(|v| v.is_real),
                    );
                    if use_temporal {
                        // Feed this track's streak and gate enforcement on
                        // fail-closed temporal confirmation.
                        let track_id = tracked.get(idx).map(|t| t.id).unwrap_or(0);
                        let entry = temporal_votes.entry(track_id).or_insert_with(|| {
                            (
                                rsface::liveness::TemporalVote::new(&temporal_cfg),
                                frame.index,
                            )
                        });
                        entry.1 = frame.index;
                        entry.0.observe(
                            live.as_ref().is_some_and(|v| v.is_real),
                            live.as_ref().map(|v| v.real_score).unwrap_or(0.0),
                        );
                        if !entry.0.confirmed() {
                            // Still warming up: not a confirmed live face.
                            blocked = self.gallery.liveness_enforce();
                        }
                        // Periodically prune streaks of tracks that left the
                        // frame so a long stream's memory stays bounded.
                        if frame.index.is_multiple_of(temporal_retire_after) {
                            temporal_votes.retain(|_, (_, last)| {
                                streak_alive(*last, frame.index, temporal_retire_after)
                            });
                        }
                    }
                    if live.as_ref().is_some_and(|v| !v.is_real) {
                        frame_spoof += 1;
                    }
                    if blocked {
                        frame_blocked += 1;
                    }
                    verdicts.push((live, blocked));
                }
                if has_face || job.kind == JobKind::Image || keepalive {
                    // 优化:keepalive 帧 + 无 face 时,annotated 与 original 字节相同。
                    // clone_rgb 是全帧 memcpy(1920x1080 RGB ~6MB),encode_png 再
                    // 跑 deflate + put S3 都白做。直接把 annotated_key 指向
                    // original_key,前端拿到的 SSE frame event annotated 字段 = 原图,
                    // 视觉上没差(本来就没框可画)。
                    // 仅在 stream keepalive 路径生效;image job 必须有独立标注层
                    // (它的 cover_key 走 original 不走 annotated,但仍要写一个)
                    // 实际上 image 也只有一张图,没有前后对比需求,可以共用。
                    if keepalive && !has_face && job.kind == JobKind::Stream {
                        result.annotated_key = result.original_key.clone();
                    } else {
                        let mut annotated = clone_rgb(&base);
                        for (d, (_, blocked)) in detections.iter().zip(verdicts.iter()) {
                            let color = if *blocked {
                                (255, 64, 64)
                            } else {
                                (0, 255, 96)
                            };
                            annotated.draw_rect(d.x, d.y, d.w, d.h, color);
                        }
                        let key = format!("jobs/{}/annotated/{:06}.png", job.id, frame.index);
                        if let Some(bytes) = encode_png(&annotated) {
                            let stored = put_with_inline_fallback(self, &key, "image/png", &bytes);
                            if let Some(b64) = inline_data_b64(&stored, &bytes) {
                                inlines.push((stored.clone(), b64));
                            }
                            result.annotated_key = Some(stored);
                        }
                    }
                }

                // 人脸裁剪。
                if has_face {
                    for (i, (d, (liveness, blocked))) in detections.iter().zip(verdicts).enumerate()
                    {
                        if total_crops >= self.cfg.max_face_crops {
                            break;
                        }
                        if let Some(crop) = crop_face(&base, d) {
                            let key = format!("jobs/{}/faces/{:06}_{i}.png", job.id, frame.index);
                            if let Some(bytes) = encode_png(&crop) {
                                let stored =
                                    put_with_inline_fallback(self, &key, "image/png", &bytes);
                                if let Some(b64) = inline_data_b64(&stored, &bytes) {
                                    inlines.push((stored.clone(), b64));
                                }
                                result.faces.push(FaceEntry {
                                    key: stored,
                                    x: d.x,
                                    y: d.y,
                                    w: d.w,
                                    h: d.h,
                                    score: d.score,
                                    liveness,
                                    blocked,
                                });
                                total_crops += 1;
                            }
                        }
                    }
                }

                job.frames
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(result.clone());
                // DB 持久化:攒 20 帧批量写一次(把每帧 1+N 次往返压成每批
                // 2 条语句);job 结束时 flush 兜底(见 loop 后),不会丢帧。
                db_pending.push(result.clone());
                if db_pending.len() >= 20 {
                    let db = self.db.clone();
                    let jid = job.id.clone();
                    let batch = std::mem::take(&mut db_pending);
                    tokio::spawn(async move {
                        db.add_frames_batch(&jid, &batch).await;
                    });
                }
                // frame 事件;若有 inline 数据,放进 `inline` 字段供前端直接渲染。
                let mut evt = serde_json::json!({"type": "frame", "frame": result});
                if !inlines.is_empty() {
                    let mut inline_obj = serde_json::Map::new();
                    for (k, v) in inlines {
                        inline_obj.insert(k, serde_json::Value::String(v));
                    }
                    evt["inline"] = serde_json::Value::Object(inline_obj);
                }
                job.emit(&evt.to_string());
            }

            {
                let mut st = job.stats.lock().unwrap_or_else(|e| e.into_inner());
                st.frames_processed += 1;
                if has_face {
                    st.frames_with_face += 1;
                }
                st.total_detections += detections.len() as u64;
                st.spoof_detections += frame_spoof;
                st.blocked_detections += frame_blocked;
                // O(1) 增量计数:per-job stats 锁已经在上面拿过,
                // 顺带 bump 平台级 MetricsAgg + per-algo detections,免得后面
                // /api/metrics /api/jobs/stats 还要扫 registry。
                // 算法名在 run_job 早期 build_detector 后已确定并迁移过桶。
                let dets_inc = detections.len() as u64;
                self.bump_frame_counters(
                    1,
                    if has_face { 1 } else { 0 },
                    dets_inc,
                );
                self.bump_liveness_counters(frame_spoof, frame_blocked);
                self.bump_algo_detections(&algo_name, dets_inc);
            }
            // 周期性把 stats 写回 DB(每 30 帧)
            if frame_idx.is_multiple_of(30) {
                let db = self.db.clone();
                let jid = job.id.clone();
                let st = job.stats.lock().unwrap_or_else(|e| e.into_inner()).clone();
                tokio::spawn(async move {
                    db.update_job_stats(&jid, &st).await;
                });
            }
            frame_idx += 1;
        }

        job.set_status(JobStatus::Done);
        {
            let db = self.db.clone();
            let id = job.id.clone();
            let st = job.stats.lock().unwrap_or_else(|e| e.into_inner()).clone();
            tokio::spawn(async move {
                db.update_job_status(
                    &id,
                    JobStatus::Done,
                    Some(crate::persist::now_ms_u64()),
                    None,
                )
                .await;
                db.update_job_stats(&id, &st).await;
            });
        }
        // 收尾 flush:不足一批的残余帧也落库(防止丢帧窗口)。
        if !db_pending.is_empty() {
            let db = self.db.clone();
            let jid = job.id.clone();
            tokio::spawn(async move {
                db.add_frames_batch(&jid, &db_pending).await;
            });
        }
        finalize(
            job,
            started,
            &self.cfg.tmp_dir,
            self,
            &algo_name,
            JobStatus::Done,
        );
        job.emit(
            &serde_json::json!({
                "type": "done",
                "stats": *job.stats.lock().unwrap_or_else(|e| e.into_inner()),
            })
            .to_string(),
        );
        Ok(())
    }
}

/// 启动死锁 janitor:每 30s 扫一次 DB,把 `status='running' AND
/// last_heartbeat_ts < now() - 30s` 的行标 error(error='worker_died')。
///
/// `shutdown` 是平台 graceful-shutdown 标志:置 true 后本任务下一轮
/// 醒来即退出,避免 shutdown 时 fire-and-forget task 撑住 axum 的
/// graceful_shutdown。
///
/// 测试 hook:
/// - `stale_secs_override` 允许集成测试把 30s 阈值压到 1s,
///   让"worker 死亡"场景不需要真的等 30 秒。`None` = 用 30s 默认。
/// - `tick_secs_override` 允许集成测试把扫描周期从 30s 压到更短,
///   让"shutdown 后 janitor 退出"测试不等 30s。`None` = 用 30s 默认。
pub fn spawn_janitor(
    db: Arc<Db>,
    shutdown: Arc<AtomicBool>,
    stale_secs_override: Option<u64>,
) -> tokio::task::JoinHandle<()> {
    spawn_janitor_with_tick(db, shutdown, stale_secs_override, None)
}

/// 与 `spawn_janitor` 同,但允许测试同时覆盖 tick 周期。生产路径
/// 不要直接调这个 — 让默认 30s 周期生效以免 janitor 抢 DB 连接。
pub fn spawn_janitor_with_tick(
    db: Arc<Db>,
    shutdown: Arc<AtomicBool>,
    stale_secs_override: Option<u64>,
    tick_secs_override: Option<u64>,
) -> tokio::task::JoinHandle<()> {
    let stale = stale_secs_override.unwrap_or(30);
    let tick_secs = tick_secs_override.unwrap_or(30);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(tick_secs));
        // 跳过首次立即触发 — server 刚启动时 running 行可能还在 mark_started
        // 路径中,last_heartbeat_ts 还没写。30s 后第一次扫描,已有 30s 的
        // 历史 running 行才会被纳入判断。
        tick.tick().await;
        loop {
            tick.tick().await;
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                tracing::info!("[janitor] shutdown signal received, exiting");
                return;
            }
            match db.janitor_kill_stale(stale as i64).await {
                Some(0) => {} // 没有 zombie,跳过日志
                Some(n) => {
                    tracing::warn!("[janitor] marked {n} stale job(s) as worker_died");
                }
                None => {
                    // DB 不可用;不要紧,起动时 reap_orphans 会兜底。
                }
            }
        }
    })
}

/// 心跳 coalescer:每 5s 一次,把 registry 内所有 `status='running'` 的
/// job 的三列时间戳(heartbeat_at / last_heartbeat_ts / updated_at)用
/// 一条 PG `WHERE id = ANY($1)` 批量刷一次。
///
/// 旧实现每个 running job 各自 spawn 一个 tokio task,每 5s 调
/// `heartbeat()` + `heartbeat_realtime()` 两条独立 UPDATE,N 个并发
/// job 就是 2N 次 / 5s DB roundtrip;CPU 上下文切换 + DB connection
/// pool 抢占都消耗可观。新实现与 N 无关,固定 1 次 / 5s。
///
/// **不替代** run_job 主循环里的 SSE 心跳:每秒的 fps 计算需要
/// per-job `frames_processed` 增量,跨 job 不好做;那条路径依然走
/// per-job tokio::spawn(只发 SSE 不写 DB)。
///
/// shutdown 同步 axum graceful_shutdown,fire-and-forget task 不会
/// 撑住 graceful_shutdown 等待。
pub fn spawn_heartbeat_coalescer(
    db: Arc<crate::persist::Db>,
    registry: Arc<JobRegistry>,
    shutdown: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // 第一次立即 tick:让 server 刚启就有 last_heartbeat_ts 落库,
        // 否则 startup-reaper + 运行 janitor 在前 5s 内看不到任何
        // "alive" 标记。后续按 5s 节奏对齐。
        let mut running_ids: Vec<String> = registry
            .list()
            .into_iter()
            .filter(|j| j.status() == JobStatus::Running)
            .map(|j| j.id.clone())
            .collect();
        if !running_ids.is_empty() {
            db.heartbeat_batch(&running_ids).await;
        }
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(5));
        tick.tick().await; // 跳过首次立即触发(上面已经手动写过)
        loop {
            tick.tick().await;
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                tracing::info!("[heartbeat] shutdown signal received, exiting");
                return;
            }
            running_ids = registry
                .list()
                .into_iter()
                .filter(|j| j.status() == JobStatus::Running)
                .map(|j| j.id.clone())
                .collect();
            if running_ids.is_empty() {
                continue;
            }
            db.heartbeat_batch(&running_ids).await;
        }
    })
}

/// 终态收尾:写 elapsed_ms + 推聚合 + 删 tmp 目录。
/// registry 引用用来同步 bump 平台级 MetricsAgg(老代码这条路径只写 stats,
/// 现在 elapsed_ms 写完之后立刻也推一份到 avg_job_ms 计数器 + per-algo 桶)。
/// elapsed_ms 是 ms 精度,Relaxed 单调累加足够。
fn finalize(
    job: &Job,
    started: std::time::Instant,
    tmp_dir: &Path,
    registry: &JobRegistry,
    algo_name: &str,
    terminal: JobStatus,
) {
    let elapsed_ms = started.elapsed().as_millis() as u64;
    job.stats
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .elapsed_ms = elapsed_ms;
    // 同步推到 agg + per-algo 桶:供 /api/metrics 的 avg_job_ms、
    // /api/jobs/stats 的 per-algo avg_elapsed_ms 使用。`finalize` 只
    // 在 run_job 主路径 (Done / Cancelled / Error) 调用一次,所以
    // 这三处增量不会重复。
    registry.record_terminal_elapsed(elapsed_ms);
    registry.record_algo_elapsed(algo_name, elapsed_ms);
    registry.finalize_algo_status(algo_name, terminal);
    // 清理临时目录(异步尽力而为)。必须用配置的 TMP_DIR,而非硬编码
    // /tmp/rsface-jobs——否则非默认部署下每个 job 的工作目录都会泄漏在磁盘上。
    let dir = tmp_dir.join(&job.id);
    std::thread::spawn(move || {
        let _ = std::fs::remove_dir_all(dir);
    });
}

/// 轮询等待同步 ffmpeg 转码子进程,期间周期性检查取消标志。
///
/// 旧代码直接 `child.wait()`,用户点"取消"/服务停机时阻塞的 wait 不会被打断,
/// 长视频转码会一直跑到结束(甚至卡死时永不退出)。这里每 200ms `try_wait`
/// 一次;一旦 `cancel` 置位就 kill 子进程、回收并返回错误,让 run_job 提前结束。
fn wait_child_cancellable(
    mut child: std::process::Child,
    cancel: &AtomicBool,
) -> std::io::Result<std::process::ExitStatus> {
    const POLL_INTERVAL_MS: u64 = 200;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if cancel.load(Ordering::SeqCst) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::other("ffmpeg transcode cancelled"));
        }
        std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
    }
}

/// 非 core 直接支持的图片统一转 PGM(灰度,无压缩)。
/// 原因:core 的 PNG 解码器只支持 stored(未压缩)块,但 ffmpeg 输出的 PNG
/// 永远是 deflate 压缩,与 core 兼容很脆弱。改用 PGM(P5)就完全规避了
/// PNG/PPM 的格式协商,平台无需触碰 core。
fn normalize_image_input(input: &str, work_dir: &Path) -> std::io::Result<String> {
    let p = Path::new(input);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        // PNG 假定 core 自己能读;只有当 core 失败时(例如 ffmpeg 出的 deflate PNG)
        // 我们才在 run_job 里兜底转 PPM。这里只对明显不识别的格式调用 ffmpeg。
        "png" | "pgm" | "ppm" => Ok(input.to_string()),
        _ => {
            // 改用 PPM(P6)保留 RGB,而不是 PGM(灰度) — 旧实现 `-pix_fmt gray`
            // 把彩色源转灰了,导致前端看到的标注/裁剪全是灰色。
            let out = work_dir.join("input.ppm");
            let status = std::process::Command::new("ffmpeg")
                .args([
                    "-y", "-i", input, "-pix_fmt", "rgb24", "-f",
                    "image2", // 强制单图输出
                ])
                .arg(&out)
                .output()
                .map_err(|e| std::io::Error::other(format!("ffmpeg spawn failed: {e}")))?;
            if !status.status.success() || !out.is_file() {
                return Err(std::io::Error::other(format!(
                    "ffmpeg image convert failed: {}",
                    String::from_utf8_lossy(&status.stderr)
                )));
            }
            Ok(out.to_string_lossy().to_string())
        }
    }
}

/// The unannotated RGB pixels of `frame`, as cheaply as possible.
///
/// `Frame::rgb` is an `Arc<RgbImage>` shared with the decoder, so when the
/// source provided colour we hand back a *second* Arc to the same buffer
/// rather than a fresh full-frame copy. The caller must treat the result as
/// read-only and clone before drawing (see the annotated-frame path).
/// Only the grayscale-source case (Arc absent) actually allocates.
fn frame_rgb(frame: &Frame) -> std::borrow::Cow<'_, RgbImage> {
    match &frame.rgb {
        Some(rgb) => std::borrow::Cow::Borrowed(rgb),
        None => std::borrow::Cow::Owned(RgbImage::from_gray(&frame.gray)),
    }
}

/// Whether a temporal streak is still retained given the frame it was last
/// seen on. A track unseen for `retire_after` frames or more is pruned.
fn streak_alive(last_seen: u64, current: u64, retire_after: u64) -> bool {
    current.saturating_sub(last_seen) < retire_after
}

fn clone_rgb(src: &RgbImage) -> RgbImage {
    let mut out = RgbImage::new(src.width(), src.height());
    out.as_mut_slice().copy_from_slice(src.as_slice());
    out
}

fn encode_png(img: &RgbImage) -> Option<Vec<u8>> {
    let mut buf = Vec::new();
    write_png_rgb(&mut buf, img).ok()?;
    Some(buf)
}

fn crop_face(base: &RgbImage, d: &Detection) -> Option<RgbImage> {
    let (w, h) = (base.width(), base.height());
    let x1 = d.x.min(w);
    let y1 = d.y.min(h);
    let x2 = (d.x + d.w).min(w);
    let y2 = (d.y + d.h).min(h);
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    let cw = x2 - x1;
    let ch = y2 - y1;
    let mut out = RgbImage::new(cw, ch);
    let src = base.as_slice();
    let dst = out.as_mut_slice();
    for row in 0..ch {
        let s_off = ((y1 + row) * w + x1) * 3;
        let d_off = row * cw * 3;
        dst[d_off..d_off + cw * 3].copy_from_slice(&src[s_off..s_off + cw * 3]);
    }
    Some(out)
}

/// 供 API 层把上传字节落盘。
/// 生成分块直落的暂存路径:`tmp_dir/staging/upload-<ms>-<seq>.part`。
/// 与 job id 解耦——慢速上传在 `create()` 占队列名额之前就先落盘,
/// 不会让一个慢客户端长期占用排队深度。
pub fn staging_upload_path(cfg: &Config) -> PathBuf {
    let n = UPLOAD_SEQ.fetch_add(1, Ordering::Relaxed);
    cfg.tmp_dir
        .join("staging")
        .join(format!("upload-{}-{n}.part", now_ms()))
}

/// 把暂存文件移动(同文件系统 rename)到 job 的工作目录,命名 `input.{ext}`。
pub fn move_staged_into_job(
    cfg: &Config,
    staged: &Path,
    job_id: &str,
    ext: &str,
) -> std::io::Result<PathBuf> {
    let dir = cfg.tmp_dir.join(job_id);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("input.{ext}"));
    std::fs::rename(staged, &path)?;
    Ok(path)
}

/// 写入媒体:S3 优先,失败则降级到本地磁盘。
/// 返回的 key 形如 `s3://<key>` 或 `local://<key>`,供 `/media/` 路由识别。
/// 保留:被单测覆盖,生产路径现在走 `put_with_fallback_file` (流式 PUT)。
#[allow(dead_code)]
fn put_with_fallback(reg: &JobRegistry, key: &str, ct: &str, bytes: &[u8]) -> String {
    match reg.s3.put_object(key, ct, bytes.to_vec()) {
        Ok(_) => format!("s3://{key}"),
        Err(e) => {
            tracing::warn!("[storage] S3 put failed for {key}: {e} — falling back to local disk");
            let path = reg.cfg.local_media_dir.join(key);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if std::fs::write(&path, bytes).is_err() {
                tracing::warn!("[storage] local write also failed for {key}");
            }
            format!("local://{key}")
        }
    }
}

/// 流式 put_with_fallback:S3 优先(走 `put_object_file`,不在内存里复制
/// 文件),失败则把磁盘文件复制到 local media(避免改源文件)。大文件时
/// 与 `put_with_fallback` 的内存峰值差异是 GB 级的。
fn put_with_fallback_file(reg: &JobRegistry, key: &str, ct: &str, src: &std::path::Path) -> String {
    match reg.s3.put_object_file(key, ct, src) {
        Ok(_) => return format!("s3://{key}"),
        Err(e) => {
            tracing::warn!("[storage] S3 stream-put failed for {key}: {e} — falling back to local")
        }
    }
    let dst = reg.cfg.local_media_dir.join(key);
    if let Some(parent) = dst.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::copy(src, &dst).is_err() {
        tracing::warn!("[storage] local stream-copy failed for {key}");
    }
    format!("local://{key}")
}

/// 同步版 put_with_fallback(从 spawn_blocking 调用,不能直接持有 S3Client 的 .put_object)。
/// 等价于 `put_with_fallback` 但不返回原始错误,直接 format 出 scheme 前缀。
pub fn put_bytes_with_fallback_blocking(
    cfg: &Config,
    key: &str,
    _ct: &str,
    bytes: &[u8],
) -> String {
    let path = cfg.local_media_dir.join(key);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&path, bytes).is_ok() {
        format!("local://{key}")
    } else {
        tracing::warn!("[storage] local write failed for {key} — returning inline://");
        format!("inline://{key}")
    }
}

/// 优雅降级版的 put:S3 + local 都失败时,返回 `inline://<key>`,表示
/// 数据已嵌入到 SSE 事件。前端拿到 `inline://` 应当读 `data_b64` 字段。
/// S3 成功或 local 成功都返回正常的 `s3://` / `local://` 前缀。
///
/// 调用方负责在 SSE `frame` 事件里把 base64 拼回去(见 `inline_data_b64`)。
fn put_with_inline_fallback(reg: &JobRegistry, key: &str, ct: &str, bytes: &[u8]) -> String {
    match reg.s3.put_object(key, ct, bytes.to_vec()) {
        Ok(_) => return format!("s3://{key}"),
        Err(e) => {
            tracing::warn!("[storage] S3 put failed for {key}: {e} — falling back to local disk");
        }
    }
    let path = reg.cfg.local_media_dir.join(key);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&path, bytes).is_ok() {
        return format!("local://{key}");
    }
    tracing::warn!("[storage] BOTH S3 and local failed for {key} — inlining base64 into SSE event");
    format!("inline://{key}")
}

/// 删除 job 在 S3 与本地媒体目录里的全部对象(prefix `jobs/{id}/`)。
/// 阻塞 IO:只能从 `spawn_blocking` 调用。S3 枚举失败时保留对象不阻断
/// 删除流程(只告警);本地目录缺失视为正常。
///
/// 中 #5:若调用方传入了 `worker`,join 一下确保后台线程退出后再列
/// prefix,防止 run_job 末尾还在 put_with_fallback(产生孤儿)。
pub fn cleanup_job_media_blocking(
    s3: &S3Client,
    cfg: &Config,
    id: &str,
    worker: Option<std::thread::JoinHandle<()>>,
) {
    if let Some(h) = worker {
        // 限定 join 等待时间,避免 cleanup 永远卡住(panic 死锁等情况)。
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        loop {
            if h.is_finished() {
                let _ = h.join();
                break;
            }
            if std::time::Instant::now() >= deadline {
                tracing::warn!("[jobs] cleanup worker join timeout for {id}, proceeding anyway");
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    let prefix = format!("jobs/{id}/");
    match s3.list_objects(&prefix) {
        Ok(keys) => {
            for key in keys {
                if let Err(e) = s3.delete_object(&key) {
                    tracing::warn!("[storage] delete object {key} failed: {e}");
                }
            }
        }
        Err(e) => {
            tracing::warn!("[storage] list {prefix} failed: {e} — orphan objects left in bucket")
        }
    }
    let local = cfg.local_media_dir.join(&prefix);
    if let Err(e) = std::fs::remove_dir_all(&local) {
        if e.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!("[storage] cleanup local {local:?} failed: {e}");
        }
    }
    // tmp 工作目录一并清掉(任务已从索引删除,没有再读的路径)。
    let tmp = cfg.tmp_dir.join(id);
    if let Err(e) = std::fs::remove_dir_all(&tmp) {
        if e.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!("[jobs] cleanup tmp {tmp:?} failed: {e}");
        }
    }
}

/// 把 `inline://` key 转成 base64 字符串(供 SSE 事件 `data_b64` 字段)。
/// `inline://` 之外的 key 一律返回 None。
fn inline_data_b64(key: &str, bytes: &[u8]) -> Option<String> {
    if key.starts_with("inline://") {
        Some(base64_encode(bytes))
    } else {
        None
    }
}

/// 极简 base64 编码(标准表,带 padding)。0-dep:不依赖 base64 crate。
fn base64_encode(input: &[u8]) -> String {
    const TBL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= input.len() {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8) | (input[i + 2] as u32);
        out.push(TBL[((n >> 18) & 0x3f) as usize] as char);
        out.push(TBL[((n >> 12) & 0x3f) as usize] as char);
        out.push(TBL[((n >> 6) & 0x3f) as usize] as char);
        out.push(TBL[(n & 0x3f) as usize] as char);
        i += 3;
    }
    let rem = input.len() - i;
    if rem == 1 {
        let n = (input[i] as u32) << 16;
        out.push(TBL[((n >> 18) & 0x3f) as usize] as char);
        out.push(TBL[((n >> 12) & 0x3f) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        out.push(TBL[((n >> 18) & 0x3f) as usize] as char);
        out.push(TBL[((n >> 12) & 0x3f) as usize] as char);
        out.push(TBL[((n >> 6) & 0x3f) as usize] as char);
        out.push('=');
    }
    out
}

/// 把 `std::panic::catch_unwind` 捕获到的 `Box<dyn Any + Send>` 转成可读字符串。
fn panic_message(info: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = info.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    if let Some(s) = info.downcast_ref::<String>() {
        return s.clone();
    }
    "<non-string panic payload>".to_string()
}

/// core 的 `source::open` 直接识别的视频后缀(走 ffmpeg pipe,内部启动 ffmpeg)。
/// 其余容器(HLS / webm / mkv / http 视频 / rtsp)由 platform 层用 ffmpeg 预处理。
fn is_core_native_video(input: &str) -> bool {
    let lower = input.to_ascii_lowercase();
    lower.ends_with(".mp4") || lower.ends_with(".mov") || lower.ends_with(".avi")
}

/// 用 ffmpeg 把任意 URL/路径转码到 work_dir 下的本地 mp4,持续写入。
/// 返回 (本地文件路径, 子进程 handle)。调用方负责 kill 子进程。
fn spawn_ffmpeg_to_local(
    input: &str,
    work_dir: &Path,
    name: &str,
) -> std::io::Result<(PathBuf, Child)> {
    let out_path = work_dir.join(name);
    let mut cmd = Command::new("ffmpeg");
    let is_stream_name = name.contains("stream");
    cmd.args([
        "-y",
        "-re",
        "-i",
        input,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "15",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-f",
        "mp4",
        "-movflags",
        if is_stream_name {
            "frag_keyframe+empty_moov+default_base_moof"
        } else {
            "+faststart"
        },
        "-flush_packets",
        "1",
    ]);
    cmd.arg(&out_path);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let child = cmd
        .spawn()
        .map_err(|e| std::io::Error::other(format!("ffmpeg spawn failed for {input}: {e}")))?;
    Ok((out_path, child))
}

/// 任务结束时把 ffmpeg 子进程收掉。
struct StreamGuard(Option<Child>);
impl Drop for StreamGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

// ---------------------------------------------------------------------------
// 检测器抽象:在平台层把 3 种 zero-dep 算法包成同一份接口,run_job 主体不分支。
// ---------------------------------------------------------------------------

/// 平台层自有的检测器枚举。
/// - `Haar`:core `Detector`(多尺度滑动窗口 + Viola-Jones 级联 + NMS)。
/// - `Cnn`: core `CnnDetector`(24×24 窗口 + CNN 前向推理 + NMS)。
/// - `Luminance`: core `LuminanceFaceDetector`(带状亮度 + 镜像对称 + 边缘密度)。
///
/// 内部使用一次构建、每次 run_job 独立持有一个实例(CnnDetector 的
/// scratch 是 !Sync,需独占单线程使用,正好匹配 run_job 的单 std::thread 模型)。
#[allow(clippy::large_enum_variant)] // 单线程构造一次,box 化不划算(856B 栈占用可接受)
pub enum DetectorKind {
    Haar(Detector),
    Cnn(CnnDetector),
    Luminance(LuminanceFaceDetector),
    Skin(SkinFaceDetector),
    Lbp(LbpFaceDetector),
    Hog(HogFaceDetector),
}

impl DetectorKind {
    /// 统一 `detect` 接口:Haar/Cnn/Luminance 各自调 core 的 detect,
    /// Cnn 先把 GrayImage → f32 `[0,1]` 缓冲,其它直接用 GrayImage。
    /// 3 个 detector 都返回 `Vec<Detection>`,run_job 不需要任何分支。
    pub fn detect(&self, gray: &GrayImage) -> Vec<Detection> {
        match self {
            DetectorKind::Haar(d) => d.detect(gray),
            DetectorKind::Cnn(d) => {
                let w = gray.width();
                let h = gray.height();
                let mut f32_img = vec![0.0f32; w * h];
                for (i, &p) in gray.as_slice().iter().enumerate() {
                    f32_img[i] = p as f32 / 255.0;
                }
                d.detect(&f32_img, w, h)
                    .into_iter()
                    .map(|cd| Detection {
                        x: cd.x,
                        y: cd.y,
                        w: cd.w,
                        h: cd.h,
                        score: cd.confidence,
                    })
                    .collect()
            }
            DetectorKind::Luminance(d) => d.detect(gray),
            DetectorKind::Skin(d) => d.detect(gray),
            DetectorKind::Lbp(d) => d.detect(gray),
            DetectorKind::Hog(d) => d.detect(gray),
        }
    }

    /// 算法名,供 SSE 事件 + /api/config + /api/jobs/{id}/compare 使用。
    /// 返回的字符串与 `KIND_*` 常量对齐 — `available_algos()` 直接消费
    /// 这套常量,避免算法列表与 kind_name 字面量各自维护。
    pub fn kind_name(&self) -> &'static str {
        match self {
            DetectorKind::Haar(_) => Self::KIND_HAAR,
            DetectorKind::Cnn(_) => Self::KIND_CNN,
            DetectorKind::Luminance(_) => Self::KIND_LUMINANCE,
            DetectorKind::Skin(_) => Self::KIND_SKIN,
            DetectorKind::Lbp(_) => Self::KIND_LBP,
            DetectorKind::Hog(_) => Self::KIND_HOG,
        }
    }

    /// 稳定的算法 ID,用于 SSE / JSON / DB 列。改这些字符串会破坏:
    /// - 前端 compare 卡片的下拉选项
    /// - `/api/config` 的 algo 字段
    /// - DB 中已有的 algo 行(`aggregate_algo_stats` 按字符串分桶)
    ///
    /// 所以是 breaking change,改名时需要双写期 + 一次性迁移。
    pub const KIND_HAAR: &'static str = "haar";
    pub const KIND_CNN: &'static str = "cnn";
    pub const KIND_LUMINANCE: &'static str = "luminance";
    pub const KIND_SKIN: &'static str = "skin";
    pub const KIND_LBP: &'static str = "lbp";
    pub const KIND_HOG: &'static str = "hog";
}

/// 解析 `RSFACE_ALGO` 环境变量,未设置时按历史规则(cnn_weights 路径
/// 或 use_cnn=true 走 cnn,否则 haar)回退。返回的字符串是 DetectorKind
/// 对应的小写名字。
fn select_algo_name(cfg: &Config) -> String {
    let from_env = std::env::var("RSFACE_ALGO")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase());
    match from_env.as_deref() {
        Some(name)
            if name == DetectorKind::KIND_HAAR
                || name == DetectorKind::KIND_CNN
                || name == DetectorKind::KIND_LUMINANCE
                || name == DetectorKind::KIND_SKIN
                || name == DetectorKind::KIND_LBP
                || name == DetectorKind::KIND_HOG =>
        {
            from_env.unwrap()
        }
        Some(other) => {
            tracing::warn!("[jobs] unknown RSFACE_ALGO='{other}', falling back to haar/cnn logic");
            if cfg.use_cnn || cfg.cnn_weights.is_some() {
                DetectorKind::KIND_CNN.to_string()
            } else {
                DetectorKind::KIND_HAAR.to_string()
            }
        }
        None => {
            if cfg.use_cnn || cfg.cnn_weights.is_some() {
                DetectorKind::KIND_CNN.to_string()
            } else {
                DetectorKind::KIND_HAAR.to_string()
            }
        }
    }
}

/// 根据 Config + `RSFACE_ALGO` 环境变量选择并构建检测器。
/// 默认行为兼容老配置:`use_cnn=true` 或 `cnn_weights` 路径已设置时走 CNN,
/// 否则走 Haar。新的 `RSFACE_ALGO` 显式覆盖以上规则,接受
/// `haar` / `cnn` / `luminance`。
///
/// `override_algo` 优先于 env 与默认(per-job 覆盖,来自 upload 表单)。
/// 必须是 `available_algos()` 之一;否则忽略。
pub fn build_detector(cfg: &Config, override_algo: Option<&str>) -> std::io::Result<DetectorKind> {
    let algo = match override_algo {
        Some(a) if available_algos().contains(&a) => a.to_string(),
        _ => select_algo_name(cfg),
    };
    match algo.as_str() {
        DetectorKind::KIND_HAAR => {
            let cascade = Cascade::load(&cfg.cascade_path).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("cascade load failed ({}): {e}", cfg.cascade_path.display()),
                )
            })?;
            let dcfg = DetectorConfig {
                min_size: cfg.min_face_size,
                use_gpu: cfg.use_gpu,
                equalize_hist: true,
                min_score: cfg.min_score,
                ..DetectorConfig::default()
            };
            Ok(DetectorKind::Haar(Detector::new(cascade, dcfg)))
        }
        DetectorKind::KIND_CNN => {
            let cnn_cfg = CnnConfig {
                window_w: cfg.min_face_size.max(8),
                window_h: cfg.min_face_size.max(8),
                ..CnnConfig::default()
            };
            let det = match &cfg.cnn_weights {
                Some(p) => {
                    let weights = CnnWeights::load(p).map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("cnn weights load failed ({}): {e}", p.display()),
                        )
                    })?;
                    CnnDetector::with_weights(weights, cnn_cfg)
                }
                None => CnnDetector::new(cnn_cfg),
            };
            Ok(DetectorKind::Cnn(det))
        }
        DetectorKind::KIND_LUMINANCE => Ok(DetectorKind::Luminance(LuminanceFaceDetector::new(
            LuminanceConfig::default(),
        ))),
        DetectorKind::KIND_SKIN => Ok(DetectorKind::Skin(SkinFaceDetector::new(
            SkinConfig::default(),
        ))),
        DetectorKind::KIND_LBP => Ok(DetectorKind::Lbp(
            LbpFaceDetector::new(LbpConfig::default()),
        )),
        DetectorKind::KIND_HOG => Ok(DetectorKind::Hog(
            HogFaceDetector::new(HogConfig::default()),
        )),
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unsupported algo '{other}'"),
        )),
    }
}

/// 给定算法名,临时构建一个对应的 DetectorKind(只为 `/compare` 端点用)。
/// 与 `build_detector` 共享选择规则,只是不依赖 cfg 里的任何路径。
pub fn build_detector_by_name(name: &str) -> std::io::Result<DetectorKind> {
    match name {
        DetectorKind::KIND_HAAR => {
            // 拿一个空的 Config 走默认 cascade 路径(平台 .env 默认 cascade.rfcf)。
            let cfg = Config::from_env();
            let cascade = Cascade::load(&cfg.cascade_path).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("cascade load failed ({}): {e}", cfg.cascade_path.display()),
                )
            })?;
            Ok(DetectorKind::Haar(Detector::new(
                cascade,
                DetectorConfig::default(),
            )))
        }
        DetectorKind::KIND_CNN => Ok(DetectorKind::Cnn(CnnDetector::new(CnnConfig::default()))),
        DetectorKind::KIND_LUMINANCE => Ok(DetectorKind::Luminance(LuminanceFaceDetector::new(
            LuminanceConfig::default(),
        ))),
        DetectorKind::KIND_SKIN => Ok(DetectorKind::Skin(SkinFaceDetector::new(
            SkinConfig::default(),
        ))),
        DetectorKind::KIND_LBP => Ok(DetectorKind::Lbp(
            LbpFaceDetector::new(LbpConfig::default()),
        )),
        DetectorKind::KIND_HOG => Ok(DetectorKind::Hog(
            HogFaceDetector::new(HogConfig::default()),
        )),
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unsupported algo '{other}'"),
        )),
    }
}

/// 列出所有可用的算法名(给 `/api/config` 和 `/compare` 用)。
///
/// 注意:历史 job 行(DB / stats)里可能仍带有已下线算法的名字
/// (`yunet` / `mtcnn` / `hog`)。这些字符串只用于展示与聚合,不经过本列表
/// 校验 —— `aggregate_algo_stats` 按原始字符串分桶,原样返回,绝不 panic。
pub fn available_algos() -> &'static [&'static str] {
    // 单一来源:与 DetectorKind::KIND_* 常量对齐,新增算法时只
    // 改 enum + 这里一处数组,避免手维护两份字符串列表导致漂移
    // (历史上 haar / cnn / luminance 的 kind_name 字面量与本列表
    // 各自维护,偶尔漏改就会出现 set_algo_override 接受但 build_detector
    // 不识别的 bug)。
    &[
        DetectorKind::KIND_HAAR,
        DetectorKind::KIND_CNN,
        DetectorKind::KIND_LUMINANCE,
        DetectorKind::KIND_SKIN,
        DetectorKind::KIND_LBP,
        DetectorKind::KIND_HOG,
    ]
}

// ---------------------------------------------------------------------------
// /api/jobs/stats:按算法聚合成功/失败/平均耗时
// ---------------------------------------------------------------------------

/// 单个算法的聚合行。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AlgoAgg {
    /// 终态 job 数(done + cancelled + error)。
    pub total: u64,
    pub done: u64,
    pub cancelled: u64,
    pub error: u64,
    /// 仍在 queued/running(不计入平均耗时)。
    pub active: u64,
    /// 终态 job 的 elapsed_ms 累计(算平均用)。
    pub elapsed_ms_sum: u64,
    /// 参与 平均耗时 的终态 job 数。
    pub timed_count: u64,
    /// 检出 face 总数。
    pub detections: u64,
}

/// 聚合输入:从 `Job` 提取的最小字段集(便于单测,不依赖整个 registry)。
#[allow(dead_code)] // 仅供单测 `aggregate_algo_stats` 喂数据用;生产路径走 algo_aggs 增量。
#[derive(Clone, Debug)]
pub struct AggSample {
    pub algo: String,
    pub status: JobStatus,
    pub elapsed_ms: u64,
    pub detections: u64,
}

/// 把一批 sample 按算法聚合成 `algo -> AlgoAgg`。
///
/// 规则:
/// - `algo` 为空的 sample(尚未构建 detector 的 queued job)归入 `"pending"`;
/// - `elapsed_ms` 只对终态(done/cancelled/error)累计,queued/running 不算;
/// - `detections` 对所有状态累计(反映实时检出量)。
#[allow(dead_code)] // 单测使用,生产路径已切到 O(1) 原子读 algo_aggs_snapshot。
pub fn aggregate_algo_stats(samples: &[AggSample]) -> std::collections::BTreeMap<String, AlgoAgg> {
    let mut out: std::collections::BTreeMap<String, AlgoAgg> = std::collections::BTreeMap::new();
    for s in samples {
        let key = if s.algo.is_empty() {
            "pending".to_string()
        } else {
            s.algo.clone()
        };
        let e = out.entry(key).or_default();
        e.detections += s.detections;
        match s.status {
            JobStatus::Done => {
                e.done += 1;
                e.total += 1;
            }
            JobStatus::Cancelled => {
                e.cancelled += 1;
                e.total += 1;
            }
            JobStatus::Error => {
                e.error += 1;
                e.total += 1;
            }
            JobStatus::Queued | JobStatus::Running => {
                e.active += 1;
            }
        }
        if is_terminal(s.status) {
            e.elapsed_ms_sum += s.elapsed_ms;
            e.timed_count += 1;
        }
    }
    out
}

impl JobRegistry {
    /// 收集 registry 内所有 job 的聚合 sample(按需调用,list 锁只短持)。
    ///
    /// **保留仅供单测用**:生产路径(`/api/jobs/stats`、`/api/metrics`)
    /// 已切到 O(1) 原子读 `algo_aggs_snapshot` / `metrics_snapshot`。
    /// 单测 `aggregate_algo_stats` 直接喂 `Vec<AggSample>`,不依赖此方法。
    #[allow(dead_code)]
    pub fn collect_agg_samples(&self) -> Vec<AggSample> {
        // 一次性把 Arc<Job> 拷出来,map 锁立刻释放;
        // 每个 job 内部的锁只在读取单 job 时短暂持有。
        self.list()
            .iter()
            .map(|j| {
                let stats = j.stats.lock().unwrap_or_else(|e| e.into_inner()).clone();
                AggSample {
                    algo: stats.algo,
                    status: j.status(),
                    elapsed_ms: stats.elapsed_ms,
                    detections: stats.total_detections,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(algo: &str, status: JobStatus, ms: u64, det: u64) -> AggSample {
        AggSample {
            algo: algo.to_string(),
            status,
            elapsed_ms: ms,
            detections: det,
        }
    }

    #[test]
    fn agg_groups_by_algo_and_counts_statuses() {
        let samples = vec![
            sample("haar", JobStatus::Done, 100, 3),
            sample("haar", JobStatus::Done, 300, 5),
            sample("haar", JobStatus::Error, 50, 0),
            sample("cnn", JobStatus::Cancelled, 200, 1),
            sample("cnn", JobStatus::Running, 999, 2),
        ];
        let agg = aggregate_algo_stats(&samples);
        let haar = &agg["haar"];
        assert_eq!(haar.total, 3);
        assert_eq!(haar.done, 2);
        assert_eq!(haar.error, 1);
        assert_eq!(haar.cancelled, 0);
        assert_eq!(haar.active, 0);
        assert_eq!(haar.elapsed_ms_sum, 450);
        assert_eq!(haar.timed_count, 3);
        assert_eq!(haar.detections, 8);

        let cnn = &agg["cnn"];
        assert_eq!(cnn.total, 1);
        assert_eq!(cnn.cancelled, 1);
        assert_eq!(cnn.active, 1);
        // running 的 999ms 不计入;cancelled 的 200ms 计入
        assert_eq!(cnn.elapsed_ms_sum, 200);
        assert_eq!(cnn.timed_count, 1);
        assert_eq!(cnn.detections, 3);
    }

    #[test]
    fn agg_empty_algo_buckets_into_pending() {
        let samples = vec![
            sample("", JobStatus::Queued, 0, 0),
            sample("", JobStatus::Done, 10, 1),
        ];
        let agg = aggregate_algo_stats(&samples);
        assert!(agg.contains_key("pending"));
        let p = &agg["pending"];
        assert_eq!(p.active, 1);
        assert_eq!(p.done, 1);
        assert_eq!(p.detections, 1);
    }

    #[test]
    fn agg_avg_ms_is_zero_when_no_terminal() {
        let samples = vec![sample("luminance", JobStatus::Running, 5000, 9)];
        let agg = aggregate_algo_stats(&samples);
        let y = &agg["luminance"];
        assert_eq!(y.timed_count, 0);
        assert_eq!(y.elapsed_ms_sum, 0);
        assert_eq!(y.active, 1);
    }

    /// 历史 DB / stats 行里可能残留已下线算法名(yunet/mtcnn/hog)。
    /// 聚合按原始字符串分桶,必须原样保留、绝不 panic 或丢数据。
    #[test]
    fn agg_tolerates_legacy_unknown_algo_strings() {
        let samples = vec![
            sample("yunet", JobStatus::Done, 100, 0),
            sample("mtcnn", JobStatus::Error, 50, 0),
            sample("hog", JobStatus::Done, 200, 0),
        ];
        let agg = aggregate_algo_stats(&samples);
        assert_eq!(agg["yunet"].total, 1);
        assert_eq!(agg["mtcnn"].error, 1);
        assert_eq!(agg["hog"].timed_count, 1);
        // 历史名字不出现在可选构造列表里,但仍能作为聚合 key 存在。
        assert!(!available_algos().contains(&"yunet"));
        assert!(agg.contains_key("yunet"));
    }

    #[test]
    fn agg_empty_input_yields_empty_map() {
        let agg = aggregate_algo_stats(&[]);
        assert!(agg.is_empty());
    }

    /// 增量聚合(O(1) 原子读)与 O(n) 扫描语义对齐测试。
    ///
    /// 模拟一个 job 的全生命周期:create → mark_started → set_status(running)
    /// → bump_frame_counters → bump_algo_detections → algo 桶迁移 →
    /// set_status(done) → finalize 路径的 record_terminal_elapsed /
    /// record_algo_elapsed / finalize_algo_status。验证:
    /// - `metrics_snapshot()` 的字段值正确
    /// - `algo_aggs_snapshot()` 的字段值正确
    /// - 路径顺序与状态机一致(running → done 单调)
    #[tokio::test]
    async fn metrics_snapshot_matches_state_machine() {
        use std::sync::Arc;
        let cfg = Config::from_env();
        let reg = Arc::new(JobRegistry {
            jobs: Mutex::new(HashMap::new()),
            s3: Arc::new(crate::s3::S3Client::new(
                "http://127.0.0.1:1".into(),
                "us-east-1".into(),
                "k".into(),
                "s".into(),
                "b".into(),
            )),
            cfg: Config {
                max_concurrent_jobs: 1,
                ..cfg
            },
            db: Arc::new(crate::persist::Db { pool: None }),
            rt: tokio::runtime::Handle::current(),
            job_slots: Arc::new(Semaphore::new(1)),
            running_jobs: AtomicU64::new(0),
            queued_jobs: AtomicU64::new(0),
            started_at: std::time::Instant::now(),
            shutdown: Arc::new(AtomicBool::new(false)),
            gallery: std::sync::Arc::new(crate::gallery::GalleryState::empty_for_tests(
                42,
                Default::default(),
            )),
            agg: Arc::new(MetricsAgg::default()),
            algo_aggs: Mutex::new(std::collections::BTreeMap::new()),
        });

        // 1) create:pending 桶 total + 1, agg 计数 queued / total_jobs +1。
        let job = reg.create(JobKind::Stream, "t".into()).unwrap();
        assert_eq!(reg.agg.total_jobs.load(Ordering::SeqCst), 1);
        assert_eq!(reg.agg.queued.load(Ordering::SeqCst), 1);
        assert_eq!(reg.agg.running.load(Ordering::SeqCst), 0);
        assert_eq!(
            reg.algo_aggs
                .lock()
                .unwrap()
                .get("pending")
                .unwrap()
                .total
                .load(Ordering::SeqCst),
            1,
            "create must bump pending.total"
        );

        // 2) mark_started:queued→running。
        reg.mark_started();
        assert_eq!(reg.agg.queued.load(Ordering::SeqCst), 0);
        assert_eq!(reg.agg.running.load(Ordering::SeqCst), 1);

        // 3) set_status(running) + 算法桶迁移 pending → haar。
        job.set_status(JobStatus::Running);
        reg.transition_algo_bucket("pending", "haar");
        let pending_after = reg
            .algo_aggs
            .lock()
            .unwrap()
            .get("pending")
            .unwrap()
            .total
            .load(Ordering::SeqCst);
        let haar_after = reg
            .algo_aggs
            .lock()
            .unwrap()
            .get("haar")
            .unwrap()
            .total
            .load(Ordering::SeqCst);
        assert_eq!(pending_after, 0, "migrate from pending must drain pending.total");
        assert_eq!(haar_after, 1, "migrate to haar must populate haar.total");

        // 4) 帧级增量。
        reg.bump_frame_counters(10, 3, 5);
        reg.bump_liveness_counters(2, 1);
        reg.bump_algo_detections("haar", 5);
        assert_eq!(reg.agg.total_frames_processed.load(Ordering::Relaxed), 10);
        assert_eq!(reg.agg.total_frames_with_face.load(Ordering::Relaxed), 3);
        assert_eq!(reg.agg.total_detections.load(Ordering::Relaxed), 5);
        assert_eq!(reg.agg.total_spoof_detections.load(Ordering::Relaxed), 2);
        assert_eq!(reg.agg.total_blocked_detections.load(Ordering::Relaxed), 1);

        // 5) 终态路径:running → done + elapsed 累加。
        //    mark_finished 在 spawn_run 的 FinishGuard::drop 里自动调,
        //    这里手动模拟一次,模拟线程退出。
        reg.mark_finished();
        job.set_status(JobStatus::Done);
        reg.record_terminal_elapsed(500);
        reg.record_algo_elapsed("haar", 500);
        reg.finalize_algo_status("haar", JobStatus::Done);
        assert_eq!(reg.agg.done.load(Ordering::SeqCst), 1);
        assert_eq!(reg.agg.running.load(Ordering::SeqCst), 0);
        assert_eq!(reg.agg.elapsed_sum_ms.load(Ordering::Relaxed), 500);
        assert_eq!(reg.agg.timed_count.load(Ordering::Relaxed), 1);

        // 6) 读侧快照:字段值与上面逐步累加结果一致。
        let snap = reg.metrics_snapshot();
        assert_eq!(snap.total_jobs, 1);
        assert_eq!(snap.running, 0);
        assert_eq!(snap.queued, 0);
        assert_eq!(snap.done, 1);
        assert_eq!(snap.errored, 0);
        assert_eq!(snap.cancelled, 0);
        assert_eq!(snap.total_frames_processed, 10);
        assert_eq!(snap.total_frames_with_face, 3);
        assert_eq!(snap.total_detections, 5);
        assert_eq!(snap.total_spoof_detections, 2);
        assert_eq!(snap.total_blocked_detections, 1);
        assert!((snap.avg_job_ms - 500.0).abs() < 0.001);

        let algo_snap = reg.algo_aggs_snapshot();
        // 桶里至少应有 haar + pending(0 total,但 lazy 创建的 entry 还在)。
        let haar = algo_snap
            .iter()
            .find(|(n, _)| n == "haar")
            .map(|(_, a)| a.clone())
            .expect("haar bucket must exist");
        assert_eq!(haar.total, 1);
        assert_eq!(haar.done, 1);
        assert_eq!(haar.active, 0, "active = total - done - cancelled - error");
        assert_eq!(haar.elapsed_ms_sum, 500);
        assert_eq!(haar.timed_count, 1);
        assert_eq!(haar.detections, 5);
    }

    /// abandoned queued job 的 pending.total 必须回退(否则历史残留)。
    #[tokio::test]
    async fn mark_abandoned_decrements_pending_total() {
        use std::sync::Arc;
        let cfg = Config::from_env();
        let reg = Arc::new(JobRegistry {
            jobs: Mutex::new(HashMap::new()),
            s3: Arc::new(crate::s3::S3Client::new(
                "http://127.0.0.1:1".into(),
                "us-east-1".into(),
                "k".into(),
                "s".into(),
                "b".into(),
            )),
            cfg: Config {
                max_concurrent_jobs: 1,
                ..cfg
            },
            db: Arc::new(crate::persist::Db { pool: None }),
            rt: tokio::runtime::Handle::current(),
            job_slots: Arc::new(Semaphore::new(1)),
            running_jobs: AtomicU64::new(0),
            queued_jobs: AtomicU64::new(0),
            started_at: std::time::Instant::now(),
            shutdown: Arc::new(AtomicBool::new(false)),
            gallery: std::sync::Arc::new(crate::gallery::GalleryState::empty_for_tests(
                42,
                Default::default(),
            )),
            agg: Arc::new(MetricsAgg::default()),
            algo_aggs: Mutex::new(std::collections::BTreeMap::new()),
        });
        let _j1 = reg.create(JobKind::Stream, "a".into()).unwrap();
        let _j2 = reg.create(JobKind::Stream, "b".into()).unwrap();
        let pending_total_before = reg
            .algo_aggs
            .lock()
            .unwrap()
            .get("pending")
            .unwrap()
            .total
            .load(Ordering::SeqCst);
        assert_eq!(pending_total_before, 2);
        // 模拟 j1 在排队阶段被取消(queued_jobs 减 1,pending.total 也减 1)。
        reg.mark_abandoned();
        let pending_total_after = reg
            .algo_aggs
            .lock()
            .unwrap()
            .get("pending")
            .unwrap()
            .total
            .load(Ordering::SeqCst);
        assert_eq!(pending_total_after, 1);
        assert_eq!(reg.agg.queued.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn streak_retention_tracks_unseen_window() {
        assert!(streak_alive(100, 100, 60), "seen this frame stays");
        assert!(streak_alive(100, 140, 60), "recently seen stays");
        assert!(
            !streak_alive(100, 160, 60),
            "exactly at the window is pruned"
        );
        assert!(!streak_alive(100, 200, 60), "long unseen is pruned");
        assert!(
            streak_alive(200, 100, 60),
            "non-monotonic input never underflows"
        );
    }

    /// 背压:queued 计数达到 max_queue_depth 时 create 返回 Err,
    /// 回退计数后可继续创建(mark_abandoned / mark_started 路径)。
    /// 用 tokio::test:create() 内部 tokio::spawn 落库任务需要 runtime。
    #[tokio::test]
    async fn queue_backpressure_rejects_when_full() {
        let cfg = Config::from_env();
        let reg = Arc::new(JobRegistry {
            jobs: Mutex::new(HashMap::new()),
            s3: Arc::new(crate::s3::S3Client::new(
                "http://127.0.0.1:1".into(),
                "us-east-1".into(),
                "k".into(),
                "s".into(),
                "b".into(),
            )),
            cfg: Config {
                max_concurrent_jobs: 1,
                max_queue_depth: 2,
                ..cfg
            },
            db: Arc::new(crate::persist::Db { pool: None }),
            rt: tokio::runtime::Handle::current(),
            job_slots: Arc::new(Semaphore::new(1)),
            running_jobs: AtomicU64::new(0),
            queued_jobs: AtomicU64::new(0),
            started_at: std::time::Instant::now(),
            shutdown: Arc::new(AtomicBool::new(false)),
            gallery: std::sync::Arc::new(crate::gallery::GalleryState::empty_for_tests(
                42,
                Default::default(),
            )),
            agg: Arc::new(MetricsAgg::default()),
            algo_aggs: Mutex::new(std::collections::BTreeMap::new()),
        });
        // 3 个 queued(permit 不消费):前 2 个 OK,第 3 个 429。
        assert!(reg.create(JobKind::Stream, "a".into()).is_ok());
        assert!(reg.create(JobKind::Stream, "b".into()).is_ok());
        assert!(reg.create(JobKind::Stream, "c".into()).is_err());
        // 回退一个 queued 计数后又可以创建。
        reg.mark_abandoned();
        assert!(reg.create(JobKind::Stream, "d".into()).is_ok());
        // 拿到 permit 转 running 后,queued 槽位释放 1 个,又能排 1 个;
        // 排满 2 后继续拒绝。
        reg.mark_started();
        assert!(reg.create(JobKind::Stream, "e".into()).is_ok());
        assert!(reg.create(JobKind::Stream, "f".into()).is_err());
        reg.mark_abandoned();
        assert!(reg.create(JobKind::Stream, "g".into()).is_ok());
        // running 扣减不死锁/不越界。
        reg.mark_finished();
        assert_eq!(reg.running_jobs.load(Ordering::SeqCst), 0);
    }

    /// 校验:`set_algo_override` 只接受 `available_algos()` 内的名字,
    /// 其它(拼写错误、未知、大小写、空)全部静默忽略。
    #[tokio::test]
    async fn algo_override_validates_against_available_algos() {
        let cfg = Config::from_env();
        let reg = Arc::new(JobRegistry {
            jobs: Mutex::new(HashMap::new()),
            s3: Arc::new(crate::s3::S3Client::new(
                "http://127.0.0.1:1".into(),
                "us-east-1".into(),
                "k".into(),
                "s".into(),
                "b".into(),
            )),
            cfg,
            db: Arc::new(crate::persist::Db { pool: None }),
            rt: tokio::runtime::Handle::current(),
            job_slots: Arc::new(Semaphore::new(2)),
            running_jobs: AtomicU64::new(0),
            queued_jobs: AtomicU64::new(0),
            started_at: std::time::Instant::now(),
            shutdown: Arc::new(AtomicBool::new(false)),
            gallery: std::sync::Arc::new(crate::gallery::GalleryState::empty_for_tests(
                42,
                Default::default(),
            )),
            agg: Arc::new(MetricsAgg::default()),
            algo_aggs: Mutex::new(std::collections::BTreeMap::new()),
        });
        let job = reg.create(JobKind::Stream, "t".into()).unwrap();
        let id = job.id.clone();
        for good in ["haar", "cnn", "luminance"] {
            reg.set_algo_override(&id, good.to_string());
            assert_eq!(
                job.algo_override
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_deref(),
                Some(good),
                "{good} must be accepted"
            );
        }
        for bad in ["", "HAAR", "yolo", "fast-rcnn", "transformer", ".."] {
            reg.set_algo_override(&id, bad.to_string());
            // 应当保持上次有效值不变(bad 不会覆盖 good)。
            let cur = job
                .algo_override
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let cur_str = cur.unwrap_or_default();
            assert!(
                available_algos().contains(&cur_str.as_str()),
                "bad input {bad:?} should not have overwritten to a non-valid algo (now = {cur_str:?})"
            );
        }
    }

    /// 校验:per-job algo override 在 `build_detector` 中优先于 env 变量。
    /// 注入 RSFACE_ALGO=haar(走 cascade)后,override=luminance 走 luminance。
    /// 这条覆盖 2026-09-08 修的 bug:之前表单 `algo` 被忽略,默认 env 路径。
    #[tokio::test]
    async fn build_detector_override_beats_env_var() {
        let cfg = Config::from_env();
        // RSFACE_ALGO 显式设 haar(env 默认就是)。override 必须成功 — luminance
        // 不需要 cascade 文件,这正是 per-job override 的核心价值。
        std::env::set_var("RSFACE_ALGO", "haar");
        let det = build_detector(&cfg, Some("luminance"))
            .expect("luminance override must succeed without cascade");
        assert_eq!(det.kind_name(), "luminance");
        std::env::remove_var("RSFACE_ALGO");
    }

    /// 校验:未知 override 字符串回退到 env(此处就是 haar),不会 panic。
    /// 这条覆盖 2026-09-08 之前用户传 `algo=foo` 时 build_detector 返回
    /// "unsupported algo 'foo'" 然后 run_job 直接挂掉的回归。
    #[tokio::test]
    async fn build_detector_unknown_override_falls_back_to_env() {
        let cfg = Config::from_env();
        std::env::set_var("RSFACE_ALGO", "haar");
        // 用 cascade.rfcf 不存在的 cfg(默认),让 fallback 也失败 → 拿到清晰错误
        // 而不是 panic。
        let res = build_detector(&cfg, Some("yolo-v8-not-supported"));
        // 失败信息应来自 cascade load(因为 fallback 仍是 haar),而不是 panic。
        let err = match res {
            Ok(_) => panic!("expected build_detector to fail with cascade.rfcf missing"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("cascade") || err.contains("algo"),
            "expected cascade or algo error, got: {err}"
        );
        std::env::remove_var("RSFACE_ALGO");
    }

    /// 校验:`begin_shutdown()` 对所有 running job 触发 cancel flag —
    /// run_job 的 frame loop 必须立即响应(不等到 watchdog 超时)。
    #[tokio::test]
    async fn begin_shutdown_signals_running_jobs() {
        let cfg = Config::from_env();
        let reg = Arc::new(JobRegistry {
            jobs: Mutex::new(HashMap::new()),
            s3: Arc::new(crate::s3::S3Client::new(
                "http://127.0.0.1:1".into(),
                "us-east-1".into(),
                "k".into(),
                "s".into(),
                "b".into(),
            )),
            cfg,
            db: Arc::new(crate::persist::Db { pool: None }),
            rt: tokio::runtime::Handle::current(),
            job_slots: Arc::new(Semaphore::new(2)),
            running_jobs: AtomicU64::new(0),
            queued_jobs: AtomicU64::new(0),
            started_at: std::time::Instant::now(),
            shutdown: Arc::new(AtomicBool::new(false)),
            gallery: std::sync::Arc::new(crate::gallery::GalleryState::empty_for_tests(
                42,
                Default::default(),
            )),
            agg: Arc::new(MetricsAgg::default()),
            algo_aggs: Mutex::new(std::collections::BTreeMap::new()),
        });
        let job = reg.create(JobKind::Stream, "long".into()).unwrap();
        let id = job.id.clone();
        reg.mark_started();
        job.set_status(JobStatus::Running);
        // cancel 还没设 → false
        assert!(!job.cancel.load(Ordering::SeqCst));
        reg.begin_shutdown();
        // 触发后 cancel = true(同 run_job 主循环检查的同一个 AtomicBool)
        assert!(
            job.cancel.load(Ordering::SeqCst),
            "begin_shutdown must flip cancel flag on every running job"
        );
        let _ = id;
    }

    #[test]
    fn face_count_locked_matches_naive_sum() {
        let frames = vec![
            FrameResult {
                index: 0,
                timestamp_ms: 0,
                annotated_key: None,
                original_key: None,
                faces: vec![FaceEntry {
                    key: "a".into(),
                    x: 0,
                    y: 0,
                    w: 1,
                    h: 1,
                    score: 1.0,
                    liveness: None,
                    blocked: false,
                }],
            },
            FrameResult {
                index: 1,
                timestamp_ms: 1,
                annotated_key: None,
                original_key: None,
                faces: vec![],
            },
        ];
        assert_eq!(face_count_locked(&frames), 1);
    }

    /// 校验:S3 不可达时,put_with_fallback 自动降级到 local:// 前缀,
    /// 不抛 panic、不丢失数据(local 文件落盘)。这是 graceful-degradation
    /// 的核心契约 — 客户端 / 前端始终能拿到一个可用的 storage key。
    #[tokio::test]
    async fn s3_unreachable_writes_local_fallback() {
        use std::sync::atomic::AtomicUsize;
        // S3 endpoint 指向黑洞地址(loopback port 1 = 永远没人在听)
        // ureq 的 connect_timeout 默认会超时;这里我们关心的是 fallback
        // 路径,不希望测试因为 S3 重试卡几秒钟 — 加一个短的 connect timeout。
        let reg = Arc::new(JobRegistry {
            jobs: Mutex::new(HashMap::new()),
            s3: Arc::new(crate::s3::S3Client::new(
                "http://127.0.0.1:1".into(),
                "us-east-1".into(),
                "k".into(),
                "s".into(),
                "b".into(),
            )),
            cfg: {
                let mut c = Config::from_env();
                c.local_media_dir = std::env::temp_dir().join("rsface-s3down-test");
                let _ = std::fs::create_dir_all(&c.local_media_dir);
                c
            },
            db: Arc::new(crate::persist::Db { pool: None }),
            rt: tokio::runtime::Handle::current(),
            job_slots: Arc::new(Semaphore::new(1)),
            running_jobs: AtomicU64::new(0),
            queued_jobs: AtomicU64::new(0),
            started_at: std::time::Instant::now(),
            shutdown: Arc::new(AtomicBool::new(false)),
            gallery: std::sync::Arc::new(crate::gallery::GalleryState::empty_for_tests(
                42,
                Default::default(),
            )),
            agg: Arc::new(MetricsAgg::default()),
            algo_aggs: Mutex::new(std::collections::BTreeMap::new()),
        });
        let key = "jobs/test-s3down/original.bin";
        let body = b"hello-s3-down".to_vec();
        // 调用会触发 S3 put (黑洞地址 → 失败),然后降级到 local disk。
        // 整个调用必须返回 Ok 不 panic。
        let storage_key = put_with_fallback(&reg, key, "application/octet-stream", &body);
        assert!(
            storage_key.starts_with("local://"),
            "expected local:// fallback when S3 unreachable, got: {storage_key}"
        );
        // 验证 local 文件确实落盘了
        let local_path = reg
            .cfg
            .local_media_dir
            .join(key.trim_start_matches("jobs/"));
        // put_with_fallback 直接 join key 到 local_media_dir,不做 trim。
        let local_path_alt = reg.cfg.local_media_dir.join(key);
        let on_disk = local_path.is_file() || local_path_alt.is_file();
        assert!(
            on_disk,
            "local fallback file missing: tried {:?} and {:?}",
            local_path, local_path_alt
        );
        // 防止未来回归:S3 端点能连上时,fallback 不应被错误触发。
        // 这里不能真连 S3(测试环境无 rustfs),只能验证前缀正确。
        let prefix_atomic = AtomicUsize::new(0); // 仅供文档用;测试体已 ok
        let _ = prefix_atomic;
    }

    /// 校验:`put_bytes_with_fallback_blocking`(被 spawn_blocking 调用,
    /// 拿不到 JobRegistry 的 arc)不依赖 S3,纯本地写入并返回 `local://` 前缀。
    /// 这条覆盖 2026-09-08 平台加固里"upload prep 阶段 S3 失败时降级"
    /// 的代码路径。
    #[tokio::test]
    async fn blocking_local_fallback_writes_file_and_returns_local_prefix() {
        let mut cfg = Config::from_env();
        let dir = std::env::temp_dir().join("rsface-blocking-test");
        let _ = std::fs::create_dir_all(&dir);
        cfg.local_media_dir = dir.clone();
        let key = "jobs/test-blocking/frame.png";
        let body = b"\x89PNG\r\n\x1a\n...stub...".to_vec();
        let result = put_bytes_with_fallback_blocking(&cfg, key, "image/png", &body);
        assert!(
            result.starts_with("local://"),
            "blocking fallback must always return local:// (no S3 dependency), got: {result}"
        );
        let on_disk = dir.join(key).is_file();
        assert!(
            on_disk,
            "blocking fallback did not create file at {:?}",
            dir.join(key)
        );
        let read_back = std::fs::read(dir.join(key)).unwrap();
        assert_eq!(read_back, body);
    }
}
