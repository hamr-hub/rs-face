//! 平台服务端入口。
//!
//! rsface-platform = rsface core(SDK)+ Web API + 任务引擎 + S3(rustfs)存储。

mod api;
mod cache;
mod config;
mod gallery;
mod gallery_handlers;
mod jobs;
mod liveness;
mod metrics;
mod persist;
mod recognition;
mod s3;
mod zip;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// 启动 tracing subscriber(env-filter 控制级别)。失败回退到默认
/// RUST_LOG=info 行为;配置错误不影响进程。
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        // 默认 info;platform 业务日志 + axum 框架的 info 都进。
        EnvFilter::new("info,rsface_platform=info,tower_http=info,axum=info")
    });
    let _ = fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time() // 时间戳在容器里由 docker/journald 加;省 IO
        .try_init();
}

#[tokio::main]
async fn main() {
    init_tracing();
    let cfg = config::Config::from_env();
    // 启动期配置校验 — 把"端口格式错/上传 0 字节/并发 0"这种会让进程
    // 跑起来但一接请求就 500 的硬错误挡在 listener bind 之前。
    if let Err(e) = config::validate(&cfg) {
        tracing::error!("[rsface-platform] FATAL config error: {e}");
        tracing::warn!("[rsface-platform] refusing to start — fix the env vars above and retry");
        std::process::exit(2);
    }
    for w in config::validate(&cfg).unwrap_or_default() {
        tracing::warn!("[rsface-platform] WARN: {w}");
    }
    let mode = if cfg.use_cnn || cfg.cnn_weights.is_some() {
        "cnn"
    } else {
        "haar"
    };
    let cnn_info = match (&cfg.cnn_weights, cfg.use_cnn) {
        (Some(p), _) => format!("weights={}", p.display()),
        (None, true) => "weights=template".to_string(),
        (None, false) => "weights=(off)".to_string(),
    };
    tracing::info!("[rsface-platform] config: bind={} s3={} bucket={} cascade={} local_dir={} db={} mode={} cnn: {} \
       concurrency: max_jobs={} queue_depth={} timeouts(image/video/stream)={}s/{}s/{}s sse_keepalive={}s parallelism={}/hint={} \
       upload_limits(image/video)={}MB/{}GB cors={} shutdown_grace={}s",
        cfg.bind_addr, cfg.s3_endpoint, cfg.s3_bucket, cfg.cascade_path.display(),
        cfg.local_media_dir.display(),
        if cfg.database_url.is_empty() { "(memory)".to_string() } else { "postgres".to_string() },
        mode, cnn_info,
        cfg.max_concurrent_jobs, cfg.max_queue_depth, cfg.job_timeout_secs, cfg.job_timeout_video_secs,
        cfg.job_timeout_stream_secs,
        cfg.sse_keepalive_secs, cfg.available_parallelism, cfg.thread_pool_hint,
        cfg.upload_limit_image / 1024 / 1024, cfg.upload_limit_video / 1024 / 1024 / 1024,
        if cfg.cors_allow_origin.is_empty() { "(off)".to_string() } else { cfg.cors_allow_origin.clone() },
        cfg.shutdown_grace_secs);

    let s3 = Arc::new(s3::S3Client::with_retry(
        cfg.s3_endpoint.clone(),
        cfg.s3_region.clone(),
        cfg.s3_access_key.clone(),
        cfg.s3_secret_key.clone(),
        cfg.s3_bucket.clone(),
        s3::RetryPolicy::from_env_counts(
            cfg.s3_max_retries,
            cfg.s3_retry_base_ms,
            cfg.s3_retry_max_ms,
        ),
    ));

    if let Err(e) = s3.ensure_bucket() {
        tracing::warn!("[rsface-platform] WARN: ensure_bucket failed: {e} (continuing; S3 may auto-create on write)");
    }
    if let Err(e) = std::fs::create_dir_all(&cfg.local_media_dir) {
        tracing::warn!("[rsface-platform] WARN: create local_media_dir failed: {e}");
    }
    // 清理上次进程残留的上传暂存文件(客户端中断/崩溃会留下 .part)。
    // 重启时不可能还有效,直接整目录清掉再重建。
    let staging_dir = cfg.tmp_dir.join("staging");
    let _ = std::fs::remove_dir_all(&staging_dir);
    if let Err(e) = std::fs::create_dir_all(&staging_dir) {
        tracing::warn!("[rsface-platform] WARN: create upload staging dir failed: {e}");
    }

    // PostgreSQL 持久化(可选;连接失败则降级为内存模式)
    let db = if !cfg.database_url.is_empty() {
        // 低 #13:DATABASE_URL 设置 → connect 失败也 fail-fast,与 migrate
        // 失败一致:对着一个连不上的 PG 跑内存模式,会静默丢全部持久化,
        // 比启动失败更危险。空字符串则保留原"纯内存"降级路径。
        let db = match persist::Db::connect_with(&cfg.database_url, cfg.database_pool_size).await {
            d if d.pool.is_none() => {
                tracing::warn!(
                    "[rsface-platform] FATAL: DATABASE_URL set but PG connect failed (set empty DATABASE_URL to run memory-only)"
                );
                std::process::exit(1);
            }
            d => d,
        };
        // 迁移失败不降级:库在但 schema 残缺时,降级会静默丢所有持久化。
        // fail-fast 让容器进入 crash loop,运维能立刻看到。
        if let Err(e) = db.migrate().await {
            tracing::error!("[rsface-platform] FATAL: database migration failed: {e}");
            tracing::error!("[rsface-platform] refusing to start against an unmigrated schema; fix the database and restart");
            std::process::exit(1);
        }
        // 孤儿任务回收:上次进程崩溃 / OOM / SIGKILL 时,可能留下
        // `status IN ('queued','running')` 的"僵尸行"。在 listen bind
        // 之前把它们标 error,避免前端永远看到"卡在 running"。
        // 阈值 5 分钟(>heartbeat 30s 间隔 × 10,确保正常 job 不会被误伤)。
        match db.reap_orphans(300).await {
            Some(n) if n > 0 => {
                println!("[rsface-platform] reaped {n} orphaned job(s) from previous run")
            }
            Some(_) => {} // 0 跳过日志
            None => eprintln!("[rsface-platform] WARN: orphan reap skipped (DB pool unavailable)"),
        }
        Arc::new(db)
    } else {
        tracing::warn!("[rsface-platform] no DATABASE_URL — running in memory-only mode");
        Arc::new(persist::Db { pool: None })
    };

    // 并发槽位:用 Semaphore 限制同时跑 job 数。permit 数 = max_concurrent_jobs。
    let job_slots = Arc::new(tokio::sync::Semaphore::new(cfg.max_concurrent_jobs));

    let state = Arc::new(jobs::JobRegistry {
        jobs: Mutex::new(HashMap::new()),
        s3: s3.clone(),
        cfg: cfg.clone(),
        db: db.clone(),
        rt: tokio::runtime::Handle::current(),
        job_slots,
        running_jobs: std::sync::atomic::AtomicU64::new(0),
        queued_jobs: std::sync::atomic::AtomicU64::new(0),
        started_at: std::time::Instant::now(),
        shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        gallery_cache: Mutex::new(None),
        gallery: Arc::new(gallery::GalleryState::load((*db).clone(), &cfg).await),
    });

    // 响应缓存层(无外部依赖)。四份 TTL cache:
    // - config 启动后不变,1h TTL
    // - metrics 每 1s 复用,前端 2s 轮询 50% 命中
    // - jobs list 500ms 复用,前端 SSE 期间频繁轮询(SSE 帧事件 + 列表刷新)70% 命中
    // - jobs stats 1s 复用,按算法聚合无状态机依赖
    let caches = api::ResponseCaches {
        config_json: Arc::new(cache::TtlCache::new(
            "config_json",
            std::time::Duration::from_secs(3600),
        )),
        metrics_json: Arc::new(cache::TtlCache::new(
            "metrics_json",
            std::time::Duration::from_millis(1000),
        )),
        jobs_list_json: Arc::new(cache::TtlCache::new(
            "jobs_list_json",
            std::time::Duration::from_millis(500),
        )),
        jobs_stats_json: Arc::new(cache::TtlCache::new(
            "jobs_stats_json",
            std::time::Duration::from_millis(1000),
        )),
    };

    let app = api::router(state.clone(), caches);

    // 死锁 janitor:周期扫描 stale running 行。DB 不可用时 janitor 静默
    // 无害(reap_orphans + heartbeat 写入路径在 DB None 时早返回)。
    // shutdown 与 axum graceful_shutdown 同步,避免拖住进程退出。
    let _janitor = jobs::spawn_janitor(db.clone(), state.shutdown.clone(), None);

    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr)
        .await
        .unwrap_or_else(|e| panic!("bind {} failed: {e}", cfg.bind_addr));
    tracing::info!("[rsface-platform] listening on http://{}", cfg.bind_addr);

    // 优雅停机三段式:
    // 1) ctrl-c / SIGTERM → axum 停止接受新连接,存量 HTTP 请求排空;
    // 2) registry.begin_shutdown():对所有非终态任务发 cancel(检测线程在
    //    下一个 frame 边界响应,长任务秒级退出,挂死的由 watchdog 超时兜底);
    // 3) 等 active 任务归零(上限 shutdown_grace_secs),给 fire-and-forget
    //    的 tokio::spawn DB 写入留出 flush 窗口,再退出进程。
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");

    tracing::info!("[rsface-platform] http drained, signalling jobs to stop");
    state.begin_shutdown();

    let grace = std::time::Duration::from_secs(cfg.shutdown_grace_secs);
    let poll = std::time::Duration::from_millis(200);
    let start = std::time::Instant::now();
    loop {
        let active = state.active_count();
        if active == 0 {
            tracing::info!("[rsface-platform] all jobs reached terminal state");
            break;
        }
        if start.elapsed() >= grace {
            tracing::warn!("[rsface-platform] shutdown grace ({}s) exceeded with {active} job(s) still active — forcing exit",
                cfg.shutdown_grace_secs);
            break;
        }
        tracing::info!("[rsface-platform] waiting for {active} active job(s) to drain ...");
        tokio::time::sleep(poll).await;
    }

    // 给已 spawn 的 DB 写入(状态/统计落库)最后一个 flush 窗口。
    // 这些任务是 fire-and-forget 的 tokio::spawn,无 join handle;
    // 主 runtime drop 前让出 500ms 足以让 PG 往返完成。
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    db.close().await;
    tracing::info!("[rsface-platform] bye");
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.ok();
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("[rsface-platform] shutting down");
}
