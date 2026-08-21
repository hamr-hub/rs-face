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
    let mode = if cfg.use_cnn || cfg.cnn_weights.is_some() { "cnn" } else { "haar" };
    let cnn_info = match (&cfg.cnn_weights, cfg.use_cnn) {
        (Some(p), _) => format!("weights={}", p.display()),
        (None, true) => "weights=template".to_string(),
        (None, false) => "weights=(off)".to_string(),
    };
    println!("[rsface-platform] config: bind={} s3={} bucket={} cascade={} local_dir={} db={} mode={} cnn: {} \
       concurrency: max_jobs={} timeouts(image/video)={}s/{}s sse_keepalive={}s parallelism={}/hint={}",
        cfg.bind_addr, cfg.s3_endpoint, cfg.s3_bucket, cfg.cascade_path.display(),
        cfg.local_media_dir.display(),
        if cfg.database_url.is_empty() { "(memory)".to_string() } else { "postgres".to_string() },
        mode, cnn_info,
        cfg.max_concurrent_jobs, cfg.job_timeout_secs, cfg.job_timeout_video_secs,
        cfg.sse_keepalive_secs, cfg.available_parallelism, cfg.thread_pool_hint);

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
    });

    let app = api::router(state);

    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr)
        .await
        .unwrap_or_else(|e| panic!("bind {} failed: {e}", cfg.bind_addr));
    println!("[rsface-platform] listening on http://{}", cfg.bind_addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
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
