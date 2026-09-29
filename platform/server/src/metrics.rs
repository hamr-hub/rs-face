//! 平台级指标聚合。
//!
//! 数据源:
//! 1. `JobRegistry::agg` — 平台级原子聚合(状态计数 + 帧统计)
//! 2. `JobRegistry::algo_aggs` — per-algo 原子聚合
//! 3. 平台进程启动时间
//!
//! 暴露:
//! - `PlatformMetrics::current()` — 平台级 KPI 快照(JSON-friendly)
//! - `to_prometheus(&PlatformMetrics)` — Prometheus 文本格式
//!
//! 设计:O(1) 原子读,无 registry 扫描、无 per-job stats lock。聚合计数
//! 在 run_job 主循环 / set_status 路径同步维护,读侧只是 12 个 atomic load。

use crate::jobs::JobRegistry;
use serde::Serialize;
use std::sync::Arc;

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
    /// 累计假脸检测数(活体判定非真)。
    pub total_spoof_detections: u64,
    /// 累计强制拦截数。
    pub total_blocked_detections: u64,
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
    /// 平台级 KPI 快照:直接读 `JobRegistry::metrics_snapshot()`,O(1)
    /// 原子读、无 registry 扫描、无 per-job stats lock。
    ///
    /// 老实现是 O(n) 扫 registry + 每 job clone stats(`u64` 数列,不算重),
    /// 但 700+ jobs 时聚合一次 ~30ms;现在固定 ~µs 级,与 job 数无关。
    pub fn current(reg: &JobRegistry) -> Self {
        reg.metrics_snapshot()
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
        "rsface_spoof_detections_total",
        "Total spoof (non-real) detections",
        m,
        &|m| m.total_spoof_detections as f64,
    );
    push_g(
        &mut out,
        "rsface_blocked_detections_total",
        "Total detections blocked by liveness enforcement",
        m,
        &|m| m.total_blocked_detections as f64,
    );
    // 下列指标在 JSON 契约中保留(前端有降级展示),但尚未在 JobStats 上
    // 建模、恒为 0,不向 Prometheus 输出空序列:gpu/cpu_levels、
    // gpu_skipped、gpu_pct、live_fps、cascade_evals/passes/pass_rate。
    // run_job 真正打点后再恢复对应 series。
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
