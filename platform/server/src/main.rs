//! 平台服务端入口。
//!
//! rsface-platform = rsface core(SDK)+ Web API + 任务引擎 + S3(rustfs)存储。

mod api;
mod config;
mod jobs;
mod persist;
mod s3;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() {
    let cfg = config::Config::from_env();
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
    println!("[rsface-platform] config: bind={} s3={} bucket={} cascade={} local_dir={} db={} mode={} cnn: {} \
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

    let s3 = Arc::new(s3::S3Client::new(
        cfg.s3_endpoint.clone(),
        cfg.s3_region.clone(),
        cfg.s3_access_key.clone(),
        cfg.s3_secret_key.clone(),
        cfg.s3_bucket.clone(),
    ));

    if let Err(e) = s3.ensure_bucket() {
        eprintln!("[rsface-platform] WARN: ensure_bucket failed: {e} (continuing; S3 may auto-create on write)");
    }
    if let Err(e) = std::fs::create_dir_all(&cfg.local_media_dir) {
        eprintln!("[rsface-platform] WARN: create local_media_dir failed: {e}");
    }

    // PostgreSQL 持久化(可选;连接失败则降级为内存模式)
    let db = if !cfg.database_url.is_empty() {
        let db = persist::Db::connect(&cfg.database_url).await;
        db.migrate().await;
        Arc::new(db)
    } else {
        eprintln!("[rsface-platform] no DATABASE_URL — running in memory-only mode");
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
        shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    });

    let app = api::router(state.clone());

    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr)
        .await
        .unwrap_or_else(|e| panic!("bind {} failed: {e}", cfg.bind_addr));
    println!("[rsface-platform] listening on http://{}", cfg.bind_addr);

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

    println!("[rsface-platform] http drained, signalling jobs to stop");
    state.begin_shutdown();

    let grace = std::time::Duration::from_secs(cfg.shutdown_grace_secs);
    let poll = std::time::Duration::from_millis(200);
    let start = std::time::Instant::now();
    loop {
        let active = state.active_count();
        if active == 0 {
            println!("[rsface-platform] all jobs reached terminal state");
            break;
        }
        if start.elapsed() >= grace {
            eprintln!("[rsface-platform] shutdown grace ({}s) exceeded with {active} job(s) still active — forcing exit",
                cfg.shutdown_grace_secs);
            break;
        }
        println!("[rsface-platform] waiting for {active} active job(s) to drain ...");
        tokio::time::sleep(poll).await;
    }

    // 给已 spawn 的 DB 写入(状态/统计落库)最后一个 flush 窗口。
    // 这些任务是 fire-and-forget 的 tokio::spawn,无 join handle;
    // 主 runtime drop 前让出 500ms 足以让 PG 往返完成。
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    db.close().await;
    println!("[rsface-platform] bye");
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
    println!("[rsface-platform] shutting down");
}
