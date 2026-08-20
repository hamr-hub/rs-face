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
}

impl Config {
    pub fn from_env() -> Self {
        // 物理并行度:仅供日志参考;运行时并发由 max_concurrent_jobs 控。
        let ap = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let pool_hint = env_or("RSFACE_THREAD_POOL", "0")
            .parse::<usize>().unwrap_or(0);
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
            stream_keepalive_period: env_or("STREAM_KEEPALIVE_PERIOD", "30").parse().unwrap_or(30),
            max_face_crops: env_or("MAX_FACE_CROPS", "2000").parse().unwrap_or(2000),
            min_face_size: env_or("MIN_FACE_SIZE", "24").parse().unwrap_or(24),
            local_media_dir: PathBuf::from(env_or("LOCAL_MEDIA_DIR", "/tmp/rsface-media")),
            database_url: env_or("DATABASE_URL", ""),
            max_concurrent_jobs: env_or("MAX_CONCURRENT_JOBS", "2")
                .parse().unwrap_or(2).max(1),
            job_timeout_secs: env_or("JOB_TIMEOUT_SECS", "0")
                .parse().unwrap_or(0),
            job_timeout_video_secs: env_or("JOB_TIMEOUT_VIDEO_SECS", "0")
                .parse().unwrap_or(0),
            sse_keepalive_secs: env_or("SSE_KEEPALIVE_SECS", "15")
                .parse().unwrap_or(15),
            use_gpu: env_or("RSFACE_USE_GPU", "1") == "1",
            // RSFACE_THREAD_POOL=0 时,自动用物理并行度(并降 1 给主线程 / tokio);
            // 显式设置时尊重 env。
            thread_pool_hint: if pool_hint == 0 { ap.saturating_sub(1).max(1) } else { pool_hint },
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
