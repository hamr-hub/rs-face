//! 平台级指标聚合。
//!
//! 数据源:
//! 1. `JobRegistry::list()` — 当前在内存里的所有 job (实时)
//! 2. `JobRegistry::get(id).summary()` — 单个 job 的累计统计
//! 3. 平台进程启动时间
//!
//! 暴露:
//! - `PlatformMetrics::current()` — 平台级 KPI 快照(JSON-friendly)
//! - `to_prometheus(&PlatformMetrics)` — Prometheus 文本格式
//!
//! 设计:无锁,纯派生计算(数据全在 JobRegistry 里)。aggregate fps
//! 通过 running jobs 的 stats.fps_window 末点取最大,避免把 SSE 流量
//! 重复搬到这里。

use crate::jobs::{JobRegistry, JobStatus};
use serde::Serialize;
use std::sync::Arc;
use std::time::SystemTime;

#[derive(Clone, Debug, Serialize)]
pub struct PlatformMetrics {
    /// 进程启动 unix ms。
    pub started_ms: u64,
    /// 现在 unix ms。
    pub now_ms: u64,
    /// 累计 job 数(含已 done / error / cancelled)。
    pub total_jobs: u64,
    /// 正在跑。
    pub running: u64,
    /// 排队。
    pub queued: u64,
    /// 完成。
    pub done: u64,
    /// 错误。
    pub errored: u64,
    /// 已取消。
    pub cancelled: u64,
    /// 累计处理帧数(跨所有 done / running job)。
    pub total_frames_processed: u64,
    /// 累计含脸帧数。
    pub total_frames_with_face: u64,
    /// 累计检测数。
    pub total_detections: u64,
    /// 累计 GPU 跑过的 pyramid level 数(per-detector 累加)。
    pub total_gpu_levels: u64,
    /// 累计 CPU 跑过的 pyramid level 数。
    pub total_cpu_levels: u64,
    /// 累计 GPU 可用但 level 像素太小被跳过的次数 — 提示调高 gpu_min_pixels。
    pub total_gpu_skipped_levels: u64,
    /// 平台级 GPU 占比 %(0..=100)。= total_gpu_levels / (total_gpu_levels+total_cpu_levels)。
    pub gpu_pct: f32,
    /// 正在跑的 job 的实时 fps 取最大值。
    pub live_fps_max: f32,
    /// 正在跑的 job 的实时 fps 平均值。
    pub live_fps_avg: f32,
    /// 累计 GPU cascade 评估数。
    pub total_cascade_evals: u64,
    /// 累计 cascade 命中数(>= min_score)。
    pub total_cascade_passes: u64,
    /// 平台级 cascade 命中率。
    pub cascade_pass_rate: f32,
    /// 平均 job 时长(已 done 的,ms)。
    pub avg_job_ms: f64,
    /// 当前并发槽位(<= max_concurrent_jobs)。
    pub concurrency: u32,
    /// 配置上限。
    pub max_concurrency: u32,
    /// PostgreSQL 是否启用。
    pub db_enabled: bool,
    /// S3 端点。
    pub s3_endpoint: String,
    /// 当前算法模式("haar"/"cnn"/...)。
    pub mode: String,
}

impl PlatformMetrics {
    pub fn current(reg: &JobRegistry) -> Self {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let started_ms = now_ms.saturating_sub(reg.started_at.elapsed().as_millis() as u64);
        let jobs = reg.list();
        let mut total = 0u64;
        let mut running = 0u64;
        let mut queued = 0u64;
        let mut done = 0u64;
        let mut err = 0u64;
        let mut cancelled = 0u64;
        let mut total_frames = 0u64;
        let mut total_faces_frames = 0u64;
        let mut total_dets = 0u64;
        // GPU / cascade / fps 细项暂未在 `JobStats` 上建模,先填 0,
        // 后续 run_job 加字段时再回来打开这些求和(代码留了骨架,见
        // `total_gpu += st.gpu_levels` 那行注释掉的位置)。
        let total_gpu: u64 = 0;
        let total_cpu: u64 = 0;
        let total_skipped: u64 = 0;
        let total_evals: u64 = 0;
        let total_passes: u64 = 0;
        let _live_fps: Vec<f32> = Vec::new();
        let mut elapsed_sum: u64 = 0;
        let mut done_count: u64 = 0;
        for j in &jobs {
            total += 1;
            match j.status() {
                JobStatus::Running => running += 1,
                JobStatus::Queued => queued += 1,
                JobStatus::Done => {
                    done += 1;
                    done_count += 1;
                }
                JobStatus::Error => err += 1,
                JobStatus::Cancelled => cancelled += 1,
            }
            let st = j.stats.lock().unwrap().clone();
            total_frames += st.frames_processed;
            total_faces_frames += st.frames_with_face;
            total_dets += st.total_detections;
            // 高级字段(JobStats 暂无),保留求和骨架:
            // total_gpu += st.gpu_levels;
            // total_cpu += st.cpu_levels;
            // total_skipped += st.gpu_skipped_levels;
            // total_evals += st.cascade_evals;
            // total_passes += st.cascade_passes;
            // if let Some(&last) = st.fps_window.last() { live_fps.push(last); }
            if st.elapsed_ms > 0 && j.status() == JobStatus::Done {
                elapsed_sum += st.elapsed_ms;
            }
        }
        let total_levels = total_gpu + total_cpu;
        let gpu_pct = if total_levels > 0 {
            (total_gpu as f32) * 100.0 / (total_levels as f32)
        } else {
            0.0
        };
        // 没有 fps_window 字段就保持 0,前端对 0 有降级显示。
        let live_fps_max = 0.0f32;
        let live_fps_avg = 0.0f32;
        let cascade_pass_rate = if total_evals > 0 {
            total_passes as f32 / total_evals as f32
        } else {
            0.0
        };
        let avg_job_ms = if done_count > 0 {
            elapsed_sum as f64 / done_count as f64
        } else {
            0.0
        };
        let max_conc = reg.cfg.max_concurrent_jobs as u32;
        // 近似并发:running 数(<=max_conc),semaphore permit 是 owned 资源,这里不查它。
        let concurrency = running.min(max_conc as u64) as u32;
        let mode = if reg.cfg.use_cnn || reg.cfg.cnn_weights.is_some() {
            "cnn".to_string()
        } else {
            "haar".to_string()
        };
        Self {
            started_ms,
            now_ms,
            total_jobs: total,
            running,
            queued,
            done,
            errored: err,
            cancelled,
            total_frames_processed: total_frames,
            total_frames_with_face: total_faces_frames,
            total_detections: total_dets,
            total_gpu_levels: total_gpu,
            total_cpu_levels: total_cpu,
            total_gpu_skipped_levels: total_skipped,
            gpu_pct,
            live_fps_max,
            live_fps_avg,
            total_cascade_evals: total_evals,
            total_cascade_passes: total_passes,
            cascade_pass_rate,
            avg_job_ms,
            concurrency,
            max_concurrency: max_conc,
            db_enabled: reg.db.pool.is_some(),
            s3_endpoint: reg.cfg.s3_endpoint.clone(),
            mode,
        }
    }
}

/// 渲染 Prometheus 文本格式(0.0.4)。注意:不要把所有时间序列都写满,
/// 只输出当前最有用的部分(平台 KPI)。per-job 序列留给上层打点。
pub fn to_prometheus(m: &PlatformMetrics) -> String {
    let mut out = String::with_capacity(2048);
    let push_g = |out: &mut String,
                  name: &str,
                  help: &str,
                  m: &PlatformMetrics,
                  f: &dyn Fn(&PlatformMetrics) -> f64| {
        out.push_str(&format!(
            "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {}\n",
            f(m)
        ));
    };
    push_g(
        &mut out,
        "rsface_jobs_total",
        "Total jobs ever seen in memory",
        m,
        &|m| m.total_jobs as f64,
    );
    push_g(
        &mut out,
        "rsface_jobs_running",
        "Currently running jobs",
        m,
        &|m| m.running as f64,
    );
    push_g(
        &mut out,
        "rsface_jobs_queued",
        "Queued jobs waiting for a permit",
        m,
        &|m| m.queued as f64,
    );
    push_g(&mut out, "rsface_jobs_done", "Completed jobs", m, &|m| {
        m.done as f64
    });
    push_g(&mut out, "rsface_jobs_error", "Failed jobs", m, &|m| {
        m.errored as f64
    });
    push_g(
        &mut out,
        "rsface_jobs_cancelled",
        "Cancelled jobs",
        m,
        &|m| m.cancelled as f64,
    );
    push_g(
        &mut out,
        "rsface_frames_processed_total",
        "Total frames processed (all jobs)",
        m,
        &|m| m.total_frames_processed as f64,
    );
    push_g(
        &mut out,
        "rsface_frames_with_face_total",
        "Total frames containing ≥1 face",
        m,
        &|m| m.total_frames_with_face as f64,
    );
    push_g(
        &mut out,
        "rsface_detections_total",
        "Total face detections",
        m,
        &|m| m.total_detections as f64,
    );
    push_g(
        &mut out,
        "rsface_gpu_levels_total",
        "Cumulative pyramid levels dispatched to GPU",
        m,
        &|m| m.total_gpu_levels as f64,
    );
    push_g(
        &mut out,
        "rsface_cpu_levels_total",
        "Cumulative pyramid levels dispatched to CPU",
        m,
        &|m| m.total_cpu_levels as f64,
    );
    push_g(&mut out, "rsface_gpu_skipped_levels_total", "Cumulative pyramid levels skipped (GPU available but image too small for GPU to be worthwhile)", m, &|m| m.total_gpu_skipped_levels as f64);
    push_g(
        &mut out,
        "rsface_gpu_pct",
        "Fraction of pyramid levels handled by GPU (0..100)",
        m,
        &|m| m.gpu_pct as f64,
    );
    push_g(
        &mut out,
        "rsface_live_fps_max",
        "Max instantaneous fps across running jobs",
        m,
        &|m| m.live_fps_max as f64,
    );
    push_g(
        &mut out,
        "rsface_live_fps_avg",
        "Mean instantaneous fps across running jobs",
        m,
        &|m| m.live_fps_avg as f64,
    );
    push_g(
        &mut out,
        "rsface_cascade_evals_total",
        "Cumulative cascade window evaluations",
        m,
        &|m| m.total_cascade_evals as f64,
    );
    push_g(
        &mut out,
        "rsface_cascade_passes_total",
        "Cumulative cascade windows passing min_score",
        m,
        &|m| m.total_cascade_passes as f64,
    );
    push_g(
        &mut out,
        "rsface_cascade_pass_rate",
        "cascade_passes / cascade_evals",
        m,
        &|m| m.cascade_pass_rate as f64,
    );
    push_g(
        &mut out,
        "rsface_avg_job_ms",
        "Average wall-time of done jobs in ms",
        m,
        &|m| m.avg_job_ms,
    );
    push_g(
        &mut out,
        "rsface_concurrency",
        "Active concurrent jobs",
        m,
        &|m| m.concurrency as f64,
    );
    push_g(
        &mut out,
        "rsface_max_concurrency",
        "Configured max concurrent jobs",
        m,
        &|m| m.max_concurrency as f64,
    );
    out
}

/// 给一个便利函数,直接给 `Arc<JobRegistry>` 调 → 文本。
pub fn render_prometheus(reg: &Arc<JobRegistry>) -> String {
    to_prometheus(&PlatformMetrics::current(reg))
}
