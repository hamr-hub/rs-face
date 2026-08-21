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
use crate::persist::Db;
use crate::s3::S3Client;
use rsface::cnn::{CnnConfig, CnnDetector, CnnWeights};
use rsface::detector::{Detection, Detector, DetectorConfig};
use rsface::face_detector::FaceDetector;
use rsface::haar::Cascade;
use rsface::image::{png::write_png_rgb, GrayImage, RgbImage};
use rsface::source::{open as open_source, Frame};
use serde::Serialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
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
    pub frames_with_face: u64,
    pub total_detections: u64,
    pub elapsed_ms: u64,
    /// 当前 job 使用的算法(haar/cnn/yunet/mtcnn/hog)。
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
    /// 检测算法名(haar/cnn/yunet/mtcnn/hog),由 run_job 在 detector 构建后写入,
    /// 摘要里暴露给前端做算法过滤 chip。
    pub algo: Mutex<Option<String>>,
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
}

impl Job {
    pub fn status(&self) -> JobStatus {
        *self.status.lock().unwrap()
    }

    pub fn set_status(&self, s: JobStatus) {
        *self.status.lock().unwrap() = s;
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
        face_count_locked(&self.frames.lock().unwrap())
    }

    pub fn summary(&self) -> serde_json::Value {
        // For video/stream jobs, prefer the first annotated frame as the cover
        // thumbnail (the original key points to .mp4 which renders as a broken
        // image in <img>). For image jobs, the original_key IS the cover.
        let frames = self.frames.lock().unwrap();
        let face_count = face_count_locked(&frames);
        let cover_key: Option<String> = match self.kind {
            JobKind::Image => self.original_media_key.lock().unwrap().clone(),
            _ => frames
                .iter()
                .find_map(|f| f.annotated_key.clone())
                .or_else(|| self.original_media_key.lock().unwrap().clone()),
        };
        let frame_count = frames.len();
        drop(frames);
        serde_json::json!({
            "id": self.id,
            "kind": self.kind,
            "display_name": self.display_name,
            "status": self.status(),
            "created_ms": self.created_ms,
            "stats": *self.stats.lock().unwrap(),
            "algo": self.algo.lock().unwrap().clone(),
            "face_count": face_count,
            "frame_count": frame_count,
            "original_key": self.original_media_key.lock().unwrap().clone(),
            "cover_key": cover_key,
            "error": self.error.lock().unwrap().clone(),
            "archived": *self.archived.lock().unwrap(),
            "original_input": self.original_input.lock().unwrap().clone(),
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
    /// 停机信号:true 后 spawn_run 不再接收新任务,运行中的收到 cancel。
    pub shutdown: Arc<AtomicBool>,
}

/// 终态判断:done/cancelled/error 视为已结束,queued/running 仍在跑。
pub fn is_terminal(status: JobStatus) -> bool {
    matches!(
        status,
        JobStatus::Done | JobStatus::Cancelled | JobStatus::Error
    )
}

/// 队列背压:排队深度达到 `MAX_QUEUE_DEPTH` 时 create 返回该错误。
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
    pub fn create(&self, kind: JobKind, display_name: String) -> Result<Arc<Job>, QueueFull> {
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
        let (tx, _rx) = broadcast::channel(256);
        let dn_for_db = display_name.clone();
        let job = Arc::new(Job {
            id: new_id(),
            kind,
            display_name,
            created_ms: now_ms(),
            status: Mutex::new(JobStatus::Queued),
            frames: Mutex::new(Vec::new()),
            stats: Mutex::new(JobStats::default()),
            algo: Mutex::new(None),
            original_media_key: Mutex::new(None),
            error: Mutex::new(None),
            archived: Mutex::new(false),
            original_input: Mutex::new(None),
            cancel: Arc::new(AtomicBool::new(false)),
            event_tx: tx,
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
    }

    /// 任务线程退出(任何终态)时调用。
    fn mark_finished(&self) {
        self.running_jobs.fetch_sub(1, Ordering::SeqCst);
    }

    /// 排队阶段即被取消:不进入 running,只回退 queued 计数。
    fn mark_abandoned(&self) {
        self.queued_jobs.fetch_sub(1, Ordering::SeqCst);
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
        println!("[jobs] shutdown: cancel signalled to non-terminal jobs (total {n})");
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
            *j.original_input.lock().unwrap() = Some(input);
        }
    }

    /// 标记 job 为已归档(侧栏默认隐藏但仍在内存/DB 中)。
    pub fn set_archived(&self, id: &str, archived: bool) -> bool {
        if let Some(j) = self.jobs.lock().unwrap().get(id).cloned() {
            *j.archived.lock().unwrap() = archived;
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

    /// 批量删除(ids → Vec<bool> 表示每个 id 是否被删)。
    pub fn remove_many(&self, ids: &[String]) -> Vec<bool> {
        let mut jobs = self.jobs.lock().unwrap();
        ids.iter().map(|id| jobs.remove(id).is_some()).collect()
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
            std::thread::spawn(move || {
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
                                eprintln!("[job {id}] watchdog: timeout after {secs}s, cancelling");
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
                        *job.error.lock().unwrap() = Some(e.to_string());
                        job.set_status(JobStatus::Error);
                        job.emit(
                            &serde_json::json!({"type": "error", "message": e.to_string()})
                                .to_string(),
                        );
                        eprintln!("[job {}] error: {e}", job.id);
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
                        *job.error.lock().unwrap() = Some(format!("panic: {msg}"));
                        job.set_status(JobStatus::Error);
                        job.emit(&serde_json::json!({"type": "error", "message": format!("panic: {msg}")}).to_string());
                        eprintln!("[job {}] PANIC: {msg}", job.id);
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
                db.update_job_status(&id, JobStatus::Running, None, None)
                    .await;
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
                    // 视频 URL:同步 ffmpeg 跑完后再喂给 core(简单可靠)。
                    let (path, mut child) = spawn_ffmpeg_to_local(input, &work_dir, "video.mp4")?;
                    let _ = child.wait();
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
        if job.kind != JobKind::Stream && job.original_media_key.lock().unwrap().is_none() {
            let ext = Path::new(&input_path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("bin")
                .to_ascii_lowercase();
            let bytes = std::fs::read(&input_path)?;
            let key = format!("jobs/{}/original.{ext}", job.id);
            let ct = match ext.as_str() {
                "png" => "image/png",
                "mp4" => "video/mp4",
                "webm" => "video/webm",
                "mov" => "video/quicktime",
                "mkv" => "video/x-matroska",
                _ => "application/octet-stream",
            };
            let stored = put_with_fallback(self, &key, ct, &bytes);
            *job.original_media_key.lock().unwrap() = Some(stored.clone());
            {
                let db = self.db.clone();
                let id = job.id.clone();
                tokio::spawn(async move {
                    db.set_original_key(&id, &stored).await;
                });
            }
        }

        // 3) 检测器:按 cfg + RSFACE_ALGO 选择 5 种算法之一(haar/cnn/yunet/mtcnn/hog)。
        //    全部对外暴露同一份 `detect(gray) -> Vec<Detection>` 接口,后续
        //    frame 循环无需分支。SSE 事件同时带 `mode` (旧) 和 `algo` (新)
        //    字段,前端新老代码都能识别。
        let detector = build_detector(&self.cfg)?;
        // 记录实际使用的算法,供 /api/jobs 摘要里的 algo 字段(算法过滤 chip 用)
        let algo_name = detector.kind_name().to_string();
        *job.algo.lock().unwrap() = Some(algo_name.clone());
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
            let mut st = job.stats.lock().unwrap();
            if st.algo.is_empty() {
                st.algo = detector.kind_name().to_string();
            }
        }
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
        let is_stream = job.kind == JobKind::Stream;
        let max_frames = match job.kind {
            JobKind::Stream => self.cfg.max_frames_stream,
            _ => self.cfg.max_frames_video,
        };
        let mut frame_idx: u64 = 0;
        let mut _frames_since_reopen: u64 = 0;
        // 批量 DB 写缓冲(20 帧一批;cancel/done 路径都会 flush)。
        let mut db_pending: Vec<FrameResult> = Vec::new();
        loop {
            if job.cancel.load(Ordering::Relaxed) {
                job.set_status(JobStatus::Cancelled);
                job.emit(&serde_json::json!({"type": "cancelled"}).to_string());
                {
                    let db = self.db.clone();
                    let id = job.id.clone();
                    let st = job.stats.lock().unwrap().clone();
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
                finalize(job, started);
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
                if has_face || job.kind == JobKind::Image || keepalive {
                    let mut annotated = clone_rgb(&base);
                    for d in &detections {
                        annotated.draw_rect(d.x, d.y, d.w, d.h, (0, 255, 96));
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

                // 人脸裁剪。
                if has_face {
                    let mut total_crops: usize = job
                        .frames
                        .lock()
                        .unwrap()
                        .iter()
                        .map(|f| f.faces.len())
                        .sum();
                    for (i, d) in detections.iter().enumerate() {
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
                                });
                                total_crops += 1;
                            }
                        }
                    }
                }

                job.frames.lock().unwrap().push(result.clone());
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
                let mut st = job.stats.lock().unwrap();
                st.frames_processed += 1;
                if has_face {
                    st.frames_with_face += 1;
                }
                st.total_detections += detections.len() as u64;
            }
            // 周期性把 stats 写回 DB(每 30 帧)
            if frame_idx.is_multiple_of(30) {
                let db = self.db.clone();
                let jid = job.id.clone();
                let st = job.stats.lock().unwrap().clone();
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
            let st = job.stats.lock().unwrap().clone();
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
        finalize(job, started);
        job.emit(
            &serde_json::json!({
                "type": "done",
                "stats": *job.stats.lock().unwrap(),
            })
            .to_string(),
        );
        Ok(())
    }
}

fn finalize(job: &Job, started: std::time::Instant) {
    job.stats.lock().unwrap().elapsed_ms = started.elapsed().as_millis() as u64;
    // 清理临时目录(异步尽力而为)。
    let dir = std::path::PathBuf::from("/tmp/rsface-jobs").join(&job.id);
    std::thread::spawn(move || {
        let _ = std::fs::remove_dir_all(dir);
    });
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

fn frame_rgb(frame: &Frame) -> RgbImage {
    match &frame.rgb {
        Some(rgb) => clone_rgb(rgb),
        None => gray_to_rgb(&frame.gray),
    }
}

fn clone_rgb(src: &RgbImage) -> RgbImage {
    let mut out = RgbImage::new(src.width(), src.height());
    out.as_mut_slice().copy_from_slice(src.as_slice());
    out
}

fn gray_to_rgb(gray: &GrayImage) -> RgbImage {
    let (w, h) = (gray.width(), gray.height());
    let mut out = RgbImage::new(w, h);
    let dst = out.as_mut_slice();
    let src = gray.as_slice();
    for (i, &v) in src.iter().enumerate() {
        dst[i * 3] = v;
        dst[i * 3 + 1] = v;
        dst[i * 3 + 2] = v;
    }
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
pub fn save_upload(
    cfg: &Config,
    job_id: &str,
    ext: &str,
    bytes: &[u8],
) -> std::io::Result<PathBuf> {
    let dir = cfg.tmp_dir.join(job_id);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("input.{ext}"));
    let mut f = std::fs::File::create(&path)?;
    f.write_all(bytes)?;
    Ok(path)
}

/// 写入媒体:S3 优先,失败则降级到本地磁盘。
/// 返回的 key 形如 `s3://<key>` 或 `local://<key>`,供 `/media/` 路由识别。
fn put_with_fallback(reg: &JobRegistry, key: &str, ct: &str, bytes: &[u8]) -> String {
    match reg.s3.put_object(key, ct, bytes.to_vec()) {
        Ok(_) => format!("s3://{key}"),
        Err(e) => {
            eprintln!("[storage] S3 put failed for {key}: {e} — falling back to local disk");
            let path = reg.cfg.local_media_dir.join(key);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if std::fs::write(&path, bytes).is_err() {
                eprintln!("[storage] local write also failed for {key}");
            }
            format!("local://{key}")
        }
    }
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
        eprintln!("[storage] local write failed for {key} — returning inline://");
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
            eprintln!("[storage] S3 put failed for {key}: {e} — falling back to local disk");
        }
    }
    let path = reg.cfg.local_media_dir.join(key);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&path, bytes).is_ok() {
        return format!("local://{key}");
    }
    eprintln!("[storage] BOTH S3 and local failed for {key} — inlining base64 into SSE event");
    format!("inline://{key}")
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
// 检测器抽象:在平台层把 5 种算法包成同一份接口,run_job 主体不分支。
// ---------------------------------------------------------------------------

/// 平台层自有的检测器枚举。
/// - `Haar`:core `Detector`(多尺度滑动窗口 + Viola-Jones 级联 + NMS)。
/// - `Cnn`: core `CnnDetector`(24×24 窗口 + CNN 前向推理 + NMS)。
/// - `Yunet`: core `YunetDetector`(5 个 anchor scale + 15-dim 输出 + NMS)。
/// - `MtCnn`: core `MtcnnDetector`(P-Net → R-Net → O-Net 三段级联)。
/// - `HogSvm`: core `HogFaceDetector`(HOG 8x8 cell + Linear SVM 64x128 窗口)。
///
/// 内部使用一次构建、每次 run_job 独立持有一个实例(CnnDetector/Yunet 的
/// scratch 是 !Sync,需独占单线程使用,正好匹配 run_job 的单 std::thread 模型)。
#[allow(clippy::large_enum_variant)] // 单线程构造一次,box 化不划算(856B 栈占用可接受)
pub enum DetectorKind {
    Haar(Detector),
    Cnn(CnnDetector),
    Yunet(rsface::yunet::YunetDetector),
    MtCnn(rsface::mtcnn::MtcnnDetector),
    HogSvm(rsface::hog_face::HogFaceDetector),
}

impl DetectorKind {
    /// 统一 `detect` 接口:Haar/Cnn/Yunet/MtCnn/HogSvm 各自调 core 的 detect,
    /// Cnn 先把 GrayImage → f32 [0,1] 缓冲,其它直接用 GrayImage。
    /// 5 个 detector 都返回 `Vec<Detection>`,run_job 不需要任何分支。
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
            DetectorKind::Yunet(d) => d.detect(gray),
            DetectorKind::MtCnn(d) => d.detect(gray),
            DetectorKind::HogSvm(d) => d.detect(gray),
        }
    }

    /// 算法名,供 SSE 事件 + /api/config + /api/jobs/{id}/compare 使用。
    pub fn kind_name(&self) -> &'static str {
        match self {
            DetectorKind::Haar(_) => "haar",
            DetectorKind::Cnn(_) => "cnn",
            DetectorKind::Yunet(_) => "yunet",
            DetectorKind::MtCnn(_) => "mtcnn",
            DetectorKind::HogSvm(_) => "hog",
        }
    }
}

/// 解析 `RSFACE_ALGO` 环境变量,未设置时按历史规则(cnn_weights 路径
/// 或 use_cnn=true 走 cnn,否则 haar)回退。返回的字符串是 DetectorKind
/// 对应的小写名字。
fn select_algo_name(cfg: &Config) -> String {
    let from_env = std::env::var("RSFACE_ALGO")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase());
    match from_env.as_deref() {
        Some("haar") | Some("cnn") | Some("yunet") | Some("mtcnn") | Some("hog") => {
            from_env.unwrap()
        }
        Some(other) => {
            eprintln!("[jobs] unknown RSFACE_ALGO='{other}', falling back to haar/cnn logic");
            if cfg.use_cnn || cfg.cnn_weights.is_some() {
                "cnn".to_string()
            } else {
                "haar".to_string()
            }
        }
        None => {
            if cfg.use_cnn || cfg.cnn_weights.is_some() {
                "cnn".to_string()
            } else {
                "haar".to_string()
            }
        }
    }
}

/// 根据 Config + `RSFACE_ALGO` 环境变量选择并构建检测器。
/// 默认行为兼容老配置:`use_cnn=true` 或 `cnn_weights` 路径已设置时走 CNN,
/// 否则走 Haar。新的 `RSFACE_ALGO` 显式覆盖以上规则,接受
/// `haar` / `cnn` / `yunet` / `mtcnn` / `hog`。
pub fn build_detector(cfg: &Config) -> std::io::Result<DetectorKind> {
    let algo = select_algo_name(cfg);
    match algo.as_str() {
        "haar" => {
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
        "cnn" => {
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
        "yunet" => Ok(DetectorKind::Yunet(rsface::yunet::YunetDetector::new(
            rsface::yunet::YunetConfig::default(),
        ))),
        "mtcnn" => Ok(DetectorKind::MtCnn(rsface::mtcnn::MtcnnDetector::new(
            rsface::mtcnn::MtcnnConfig::default(),
        ))),
        "hog" => Ok(DetectorKind::HogSvm(
            rsface::hog_face::HogFaceDetector::new(rsface::hog_face::HogConfig::default()),
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
        "haar" => {
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
        "cnn" => Ok(DetectorKind::Cnn(CnnDetector::new(CnnConfig::default()))),
        "yunet" => Ok(DetectorKind::Yunet(rsface::yunet::YunetDetector::new(
            rsface::yunet::YunetConfig::default(),
        ))),
        "mtcnn" => Ok(DetectorKind::MtCnn(rsface::mtcnn::MtcnnDetector::new(
            rsface::mtcnn::MtcnnConfig::default(),
        ))),
        "hog" => Ok(DetectorKind::HogSvm(
            rsface::hog_face::HogFaceDetector::new(rsface::hog_face::HogConfig::default()),
        )),
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unsupported algo '{other}'"),
        )),
    }
}

/// 列出所有可用的算法名(给 `/api/config` 和 `/compare` 用)。
pub fn available_algos() -> &'static [&'static str] {
    &["haar", "cnn", "yunet", "mtcnn", "hog"]
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
    pub fn collect_agg_samples(&self) -> Vec<AggSample> {
        // 一次性把 Arc<Job> 拷出来,map 锁立刻释放;
        // 每个 job 内部的锁只在读取单 job 时短暂持有。
        self.list()
            .iter()
            .map(|j| {
                let stats = j.stats.lock().unwrap().clone();
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
        let samples = vec![sample("yunet", JobStatus::Running, 5000, 9)];
        let agg = aggregate_algo_stats(&samples);
        let y = &agg["yunet"];
        assert_eq!(y.timed_count, 0);
        assert_eq!(y.elapsed_ms_sum, 0);
        assert_eq!(y.active, 1);
    }

    #[test]
    fn agg_empty_input_yields_empty_map() {
        let agg = aggregate_algo_stats(&[]);
        assert!(agg.is_empty());
    }

    /// 背压:queued 计数达到 max_queue_depth 时 create 返回 Err,
    /// 回退计数后可继续创建(mark_abandoned / mark_started 路径)。
    /// 用 tokio::test:create() 内部 tokio::spawn 落库任务需要 runtime。
    #[tokio::test]
    async fn queue_backpressure_rejects_when_full() {
        let cfg = Config::from_env();
        let reg = JobRegistry {
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
            shutdown: Arc::new(AtomicBool::new(false)),
        };
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
}
