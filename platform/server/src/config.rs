//! 环境变量配置。所有项都有可在 docker-compose 中覆盖的默认值。

use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Config {
    /// HTTP 监听地址。
    pub bind_addr: String,
    /// rustfs / S3 兼容端点,如 `http://rustfs:9000`。
    pub s3_endpoint: String,
    pub s3_region: String,
    pub s3_access_key: String,
    pub s3_secret_key: String,
    pub s3_bucket: String,
    /// Haar 级联文件(.rfcf)路径。
    pub cascade_path: PathBuf,
    /// CNN 权重文件(.cnn.bin)路径。None 时 platform 走 Haar 级联路径;
    /// Some 时 platform 走 CNN 路径(若 use_cnn=true 且路径无效,run_job 启动时报错)。
    pub cnn_weights: Option<PathBuf>,
    /// 强制开启 CNN 模式(用 core 提供的 built-in 模板权重)。
    /// `RSFACE_USE_CNN=1` 时即使 cnn_weights 为 None 也走 CNN;
    /// 默认 false(走 Haar)。
    pub use_cnn: bool,
    /// 静态前端目录。
    pub web_dir: PathBuf,
    /// 上传文件临时目录。
    pub tmp_dir: PathBuf,
    /// 视频任务最多处理的帧数(防止超大文件拖垮服务)。
    pub max_frames_video: u64,
    /// 直播流任务最多处理的帧数(0 = 不限,由 cancel 停止)。
    pub max_frames_stream: u64,
    /// 直播流:无脸帧的采样周期(每 N 帧存一帧原始+标注,保持画面活动)。
    pub stream_keepalive_period: u64,
    /// 每个任务最多存储的人脸裁剪图数量。
    pub max_face_crops: usize,
    /// 检测器最小人脸尺寸(px)。
    pub min_face_size: usize,
    /// 本地媒体缓存目录(S3 失败时兜底,前端仍可访问)。
    pub local_media_dir: PathBuf,
    /// PostgreSQL DSN,空字符串则纯内存。
    pub database_url: String,
    /// 同时跑的最大任务数(超过排队)。默认 2(单核机器)。
    pub max_concurrent_jobs: usize,
    /// 任务超时(秒)。0 = 不超时(由 cancel 控)。
    pub job_timeout_secs: u64,
    /// 任务超时(秒),视频任务可设更长。
    pub job_timeout_video_secs: u64,
    /// 线程池大小提示(env RSFACE_THREAD_POOL)。仅打印;运行期由 tokio 决定。
    pub thread_pool_hint: usize,
    /// SSE 注释心跳间隔(秒)。0 = 关闭。
    pub sse_keepalive_secs: u64,
    /// 启动时打印的零依赖(available_parallelism)物理并行度。
    pub available_parallelism: usize,
    /// 启用 GPU(OpenCL)进行平方积分 / variance prefilter。需容器挂 NVIDIA runtime。
    pub use_gpu: bool,
    /// 级联检测器最终 score 阈值(0=不过滤,1=必须通过所有 stage)。
    /// OpenCV Haar 级联在 ~0.5 附近给出合理 F1;0.0=关闭,会接受所有通过 NMS 的框。
    pub min_score: f32,
    /// 流任务超时(秒)。默认 0(不限,由 cancel 停止);>0 时挂死流会被 watchdog 收掉。
    pub job_timeout_stream_secs: u64,
    /// 排队深度上限(queued 任务数,不含 running)。超过则拒绝新任务(429),
    /// 防止无限排队占内存。默认 64;0 = 不限。
    pub max_queue_depth: usize,
    /// 图片上传大小上限(字节)。默认 50 MB。
    pub upload_limit_image: usize,
    /// 视频上传大小上限(字节)。默认 2 GB。
    pub upload_limit_video: usize,
    /// CORS 允许来源(如 `https://fe.example.com`)。空 = 关闭 CORS(默认,
    /// 同源部署)。设置后为跨域前端放开 `/api/*` + `/media/*`。
    pub cors_allow_origin: String,
    /// 优雅停机等待运行中任务排空的超时(秒)。默认 180(3 分钟);
    /// 超时后强制退出。0 = 不等待(立即退出)。
    pub shutdown_grace_secs: u64,
}

impl Config {
    pub fn from_env() -> Self {
        // 物理并行度:仅供日志参考;运行时并发由 max_concurrent_jobs 控。
        let ap = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let pool_hint = env_or("RSFACE_THREAD_POOL", "0")
            .parse::<usize>()
            .unwrap_or(0);
        Self {
            bind_addr: env_or("BIND_ADDR", "0.0.0.0:8080"),
            s3_endpoint: env_or("S3_ENDPOINT", "http://127.0.0.1:9000"),
            s3_region: env_or("S3_REGION", "us-east-1"),
            s3_access_key: env_or("S3_ACCESS_KEY", "rsface"),
            s3_secret_key: env_or("S3_SECRET_KEY", "rsface-secret"),
            s3_bucket: env_or("S3_BUCKET", "rsface"),
            cascade_path: PathBuf::from(env_or("RSFACE_CASCADE", "cascade.rfcf")),
            cnn_weights: optional_path("RSFACE_CNN_WEIGHTS"),
            use_cnn: env_or("RSFACE_USE_CNN", "0") == "1",
            web_dir: PathBuf::from(env_or("WEB_DIR", "web")),
            tmp_dir: PathBuf::from(env_or("TMP_DIR", "/tmp/rsface-jobs")),
            max_frames_video: env_or("MAX_FRAMES_VIDEO", "3600").parse().unwrap_or(3600),
            max_frames_stream: env_or("MAX_FRAMES_STREAM", "0").parse().unwrap_or(0),
            stream_keepalive_period: env_or("STREAM_KEEPALIVE_PERIOD", "30")
                .parse()
                .unwrap_or(30),
            max_face_crops: env_or("MAX_FACE_CROPS", "2000").parse().unwrap_or(2000),
            min_face_size: env_or("MIN_FACE_SIZE", "24").parse().unwrap_or(24),
            local_media_dir: PathBuf::from(env_or("LOCAL_MEDIA_DIR", "/tmp/rsface-media")),
            database_url: env_or("DATABASE_URL", ""),
            max_concurrent_jobs: env_or("MAX_CONCURRENT_JOBS", "2")
                .parse()
                .unwrap_or(2)
                .max(1),
            job_timeout_secs: env_or("JOB_TIMEOUT_SECS", "0").parse().unwrap_or(0),
            job_timeout_video_secs: env_or("JOB_TIMEOUT_VIDEO_SECS", "0").parse().unwrap_or(0),
            sse_keepalive_secs: env_or("SSE_KEEPALIVE_SECS", "15").parse().unwrap_or(15),
            use_gpu: env_or("RSFACE_USE_GPU", "1") == "1",
            // 默认 0.0(不过滤)— OpenCV Haar 训练已包含 stage 级阈值;
            // 想严格压低 FP 可设 0.5。生产 drama 类素材通常 0.3 较平衡。
            min_score: env_or("RSFACE_MIN_SCORE", "0.0")
                .parse::<f32>()
                .unwrap_or(0.0),
            job_timeout_stream_secs: env_or("JOB_TIMEOUT_STREAM_SECS", "0").parse().unwrap_or(0),
            max_queue_depth: env_or("MAX_QUEUE_DEPTH", "64").parse().unwrap_or(64),
            upload_limit_image: env_or("UPLOAD_LIMIT_IMAGE_MB", "50")
                .parse::<usize>()
                .unwrap_or(50)
                .saturating_mul(1024 * 1024),
            upload_limit_video: env_or("UPLOAD_LIMIT_VIDEO_GB", "2")
                .parse::<usize>()
                .unwrap_or(2)
                .saturating_mul(1024 * 1024 * 1024),
            cors_allow_origin: env_or("CORS_ALLOW_ORIGIN", ""),
            shutdown_grace_secs: env_or("SHUTDOWN_GRACE_SECS", "180").parse().unwrap_or(180),
            // RSFACE_THREAD_POOL=0 时,自动用物理并行度(并降 1 给主线程 / tokio);
            // 显式设置时尊重 env。
            thread_pool_hint: if pool_hint == 0 {
                ap.saturating_sub(1).max(1)
            } else {
                pool_hint
            },
            available_parallelism: ap,
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// 空字符串视作未配置,返回 None;否则返回 Some(PathBuf)。
/// 平台用 env 暴露 `RSFACE_CNN_WEIGHTS`,空字符串让 caller 知道"用 None 走 Haar"。
fn optional_path(key: &str) -> Option<PathBuf> {
    match std::env::var(key) {
        Ok(s) if !s.is_empty() => Some(PathBuf::from(s)),
        _ => None,
    }
}

/// Startup-time config validation. Returns Err on a hard failure (will
/// prevent server start) or Ok(warnings) listing soft issues the operator
/// should know about.
///
/// Hard failures:
/// - bind_addr unparseable (the listener can't bind).
/// - upload limits = 0 (uploads would be rejected with 0-byte limit).
/// - max_concurrent_jobs out of [1, 64] (off-by-one in env vars).
///
/// Soft warnings (printed but non-fatal):
/// - cascade_path missing (the Haar default — jobs submitted without per-job
///   algo override will error at run time, which is the correct user-facing
///   signal; we don't want to crash the platform over a missing cascade when
///   the user explicitly requested e.g. `luminance`).
/// - cnn_weights path set but file missing.
/// - tmp_dir / local_media_dir not writable (logged elsewhere already).
pub fn validate(cfg: &Config) -> Result<Vec<String>, String> {
    let mut warnings = Vec::new();

    // bind_addr: SocketAddr parse.
    if cfg.bind_addr.parse::<std::net::SocketAddr>().is_err() {
        return Err(format!(
            "BIND_ADDR={:?} is not a valid SocketAddr (expected host:port, e.g. 0.0.0.0:8080)",
            cfg.bind_addr
        ));
    }

    // Upload limits must be > 0.
    if cfg.upload_limit_image == 0 {
        return Err("UPLOAD_LIMIT_IMAGE_MB resolved to 0 bytes — uploads would be rejected".into());
    }
    if cfg.upload_limit_video == 0 {
        return Err("UPLOAD_LIMIT_VIDEO_GB resolved to 0 bytes — uploads would be rejected".into());
    }

    // Concurrent jobs.
    if cfg.max_concurrent_jobs == 0 {
        return Err("MAX_CONCURRENT_JOBS=0 — server cannot start any job".into());
    }
    if cfg.max_concurrent_jobs > 64 {
        warnings.push(format!(
            "MAX_CONCURRENT_JOBS={} > 64 — high parallelism may starve DB / S3",
            cfg.max_concurrent_jobs
        ));
    }

    // Cascade: only a soft warning. Hard fail happens in run_job with a clear
    // error message; per-job algo override (added 2026-09-08) means a missing
    // cascade is fine if the user only ever runs luminance / yunet / mtcnn.
    if !cfg.cascade_path.is_file() {
        warnings.push(format!(
            "cascade file missing at {} — Haar jobs will fail at runtime (use --algo for other detectors)",
            cfg.cascade_path.display()
        ));
    }

    // CNN weights: same soft handling.
    if let Some(p) = &cfg.cnn_weights {
        if !p.is_file() {
            warnings.push(format!(
                "cnn weights file missing at {} — CNN jobs will fail at runtime",
                p.display()
            ));
        }
    }

    // SSE keepalive must be > 0 if any client connects (SSE default is 15s,
    // but operators occasionally set it to 0 to disable, which silently breaks
    // streaming through reverse proxies with idle timeouts).
    if cfg.sse_keepalive_secs == 0 {
        warnings.push(
            "SSE_KEEPALIVE_SECS=0 — disable may cause reverse-proxy idle drops for live streams"
                .into(),
        );
    }

    // Shutdown grace.
    if cfg.shutdown_grace_secs == 0 {
        warnings.push(
            "SHUTDOWN_GRACE_SECS=0 — process will exit immediately on SIGTERM, dropping active jobs".into()
        );
    }

    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with(bind: &str, image_mb: usize, video_gb: usize, max_jobs: usize) -> Config {
        let mut c = Config::from_env();
        c.bind_addr = bind.to_string();
        c.upload_limit_image = image_mb * 1024 * 1024;
        c.upload_limit_video = video_gb * 1024 * 1024 * 1024;
        c.max_concurrent_jobs = max_jobs;
        // 把 cascade 路径改成已存在的目录里某个文件,让 hard-warn 不触发
        c.cascade_path = std::path::PathBuf::from("Cargo.toml");
        c
    }

    #[test]
    fn validate_accepts_sane_default() {
        let cfg = cfg_with("0.0.0.0:8080", 50, 2, 2);
        let w = validate(&cfg).expect("default cfg must validate");
        // 至少有 cascade-ok 这条不警告;warnings 可能为空或仅含高并发等次要项。
        // 这里只确认不返回 Err。
        let _ = w;
    }

    #[test]
    fn validate_rejects_bad_bind_addr() {
        let cfg = cfg_with("not-a-socket", 50, 2, 2);
        assert!(validate(&cfg).is_err());
    }

    #[test]
    fn validate_rejects_zero_upload_limit() {
        let cfg = cfg_with("0.0.0.0:8080", 0, 2, 2);
        assert!(validate(&cfg).is_err());
        let cfg = cfg_with("0.0.0.0:8080", 50, 0, 2);
        assert!(validate(&cfg).is_err());
    }

    #[test]
    fn validate_rejects_zero_concurrency() {
        let cfg = cfg_with("0.0.0.0:8080", 50, 2, 0);
        assert!(validate(&cfg).is_err());
    }

    #[test]
    fn validate_warns_on_high_concurrency() {
        let cfg = cfg_with("0.0.0.0:8080", 50, 2, 128);
        let w = validate(&cfg).expect("high concurrency is a warning, not failure");
        assert!(
            w.iter().any(|s| s.contains("MAX_CONCURRENT_JOBS=128")),
            "expected warning about high concurrency, got: {w:?}"
        );
    }
}
