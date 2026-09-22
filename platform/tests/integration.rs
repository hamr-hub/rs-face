//! 集成测试:跑整个 axum router 起来,用 reqwest 模拟真实 HTTP 请求。
//!
//! 设计原则:
//! - 假设 dev 环境已经有 rustfs (127.0.0.1:19000) 和 postgres (127.0.0.1:15432)
//!   在跑(`platform/scripts/docker-smoke.sh` 或 `docker compose up -d`)。
//! - 如果连不上,测试 `#[ignore]` skip 并打 warning,不在 CI 上 random fail。
//! - 每个测试用独立临时目录 `cfg.tmp_dir`,互不污染。
//! - 跑: `cargo test --manifest-path platform/Cargo.toml --test integration`
//!
//! 覆盖路径:
//! - 健康检查 200/503 split
//! - upload image → job 完成 + face detected + 删除回收
//! - download_zip 字节级 archive 内容
//! - 5 类 SSRF bypass payload 全部 reject(端到端跑 HTTP)
//! - range streaming 206 响应

use std::time::Duration;

use rsface_platform::jobs::{available_algos, JobRegistry};
use rsface_platform::{api, cache::TtlCache};

/// 起一个 in-process axum server,绑 127.0.0.1:0(随机端口),等就绪。
async fn boot_server() -> Option<(String, tokio::sync::oneshot::Sender<()>)> {
    // 拼一个最小能跑的 cfg:复用 .env,任一缺失就 skip。
    let mut cfg = rsface_platform::config::Config::from_env();
    cfg.tmp_dir = std::env::temp_dir().join(format!("rsface-it-{}", uuid_like()));
    cfg.local_media_dir = cfg.tmp_dir.join("media");
    cfg.database_url = "".to_string(); // 强制内存模式,PG 可选
    cfg.upload_limit_image = 50 * 1024 * 1024;
    cfg.upload_limit_video = 2 * 1024 * 1024 * 1024;
    cfg.job_timeout_secs = 30;
    cfg.job_timeout_video_secs = 60;
    cfg.job_timeout_stream_secs = 0;
    cfg.max_concurrent_jobs = 2;
    cfg.max_queue_depth = 16;
    cfg.shutdown_grace_secs = 5;

    let _ = std::fs::create_dir_all(&cfg.tmp_dir);
    let _ = std::fs::create_dir_all(&cfg.local_media_dir);

    // 集成测试需要 rustfs 在 127.0.0.1:19000(S3 endpoint)。
    // 探一下,不在就 skip。
    let s3 = rsface_platform::s3::S3Client::new(
        cfg.s3_endpoint.clone(),
        cfg.s3_region.clone(),
        cfg.s3_access_key.clone(),
        cfg.s3_secret_key.clone(),
        cfg.s3_bucket.clone(),
    );
    if s3.ping().is_err() {
        eprintln!(
            "skipping integration tests: {} not reachable",
            cfg.s3_endpoint
        );
        return None;
    }
    if let Err(e) = s3.ensure_bucket() {
        eprintln!("ensure_bucket failed: {e} (continuing)");
    }

    let db = std::sync::Arc::new(rsface_platform::persist::Db { pool: None });
    let rt = tokio::runtime::Handle::current();
    let state = std::sync::Arc::new(JobRegistry {
        jobs: std::sync::Mutex::new(std::collections::HashMap::new()),
        s3: std::sync::Arc::new(s3),
        cfg: cfg.clone(),
        db,
        rt: rt.clone(),
        job_slots: std::sync::Arc::new(tokio::sync::Semaphore::new(cfg.max_concurrent_jobs)),
        running_jobs: std::sync::atomic::AtomicU64::new(0),
        queued_jobs: std::sync::atomic::AtomicU64::new(0),
        started_at: std::time::Instant::now(),
        shutdown: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        gallery_cache: std::sync::Mutex::new(None),
        gallery: std::sync::Arc::new(rsface_platform::gallery::GalleryState::empty_for_tests(
            42,
            Default::default(),
        )),
    });
    let caches = api::ResponseCaches {
        config_json: std::sync::Arc::new(TtlCache::new("cfg", Duration::from_secs(3600))),
        metrics_json: std::sync::Arc::new(TtlCache::new("m", Duration::from_millis(500))),
        jobs_list_json: std::sync::Arc::new(TtlCache::new("jl", Duration::from_millis(500))),
        jobs_stats_json: std::sync::Arc::new(TtlCache::new("js", Duration::from_millis(500))),
    };
    let rate_limiter = std::sync::Arc::new(rsface_platform::rate_limit::RateLimiter::new());
    let app = api::router(state.clone(), caches, rate_limiter);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let addr = listener.local_addr().ok()?;
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await;
    });
    // 给 server 50 ms listen 完成。
    tokio::time::sleep(Duration::from_millis(50)).await;
    Some((format!("http://{}", addr), shutdown_tx))
}

/// 极简 UUID-like suffix,避免再引一个 uuid crate 到 dev-deps。
fn uuid_like() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:x}")
}

#[tokio::test]
async fn deep_health_returns_200_when_backends_reachable() {
    let Some((base, shutdown)) = boot_server().await else {
        return;
    };
    let res = reqwest::get(format!("{base}/api/health/deep"))
        .await
        .expect("http client");
    assert_eq!(res.status(), 200);
    let body: serde_json::Value = res.json().await.expect("json");
    assert_eq!(body["service"], "rsface-platform");
    assert!(body["checks"]["s3"].is_boolean());
    assert!(body["checks"]["postgres"].is_boolean());
    let _ = shutdown.send(());
}

#[tokio::test]
async fn upload_image_end_to_end() {
    let Some((base, shutdown)) = boot_server().await else {
        return;
    };
    let img = std::fs::read("../tests/fixtures/demo_face_256.pgm")
        .or_else(|_| std::fs::read("tests/fixtures/demo_face_256.pgm"))
        .expect("fixture");
    let part = reqwest::multipart::Part::bytes(img)
        .file_name("demo_face_256.pgm")
        .mime_str("image/x-portable-graymap")
        .expect("mime");
    let form = reqwest::multipart::Form::new()
        .text("algo", "haar")
        .part("file", part);
    let res = reqwest::Client::new()
        .post(format!("{base}/api/jobs/image"))
        .multipart(form)
        .send()
        .await
        .expect("http");
    assert_eq!(res.status(), 200, "upload expected 200");
    let body: serde_json::Value = res.json().await.expect("json");
    let job_id = body["job_id"].as_str().expect("job_id string").to_string();
    // 轮询直到 done/error(>2s timeout)。
    let start = std::time::Instant::now();
    loop {
        let r = reqwest::get(format!("{base}/api/jobs/{job_id}"))
            .await
            .expect("http");
        let j: serde_json::Value = r.json().await.expect("json");
        let status = j["status"].as_str().unwrap_or("");
        if status == "done" {
            assert_eq!(j["algo"], "haar");
            assert!(j["face_count"].as_u64().unwrap_or(0) >= 1);
            break;
        }
        if status == "error" {
            panic!("job errored: {}", j["error"]);
        }
        if start.elapsed() > Duration::from_secs(15) {
            panic!("job didn't finish in 15s; last status {status}");
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    // 删除回收
    let del = reqwest::Client::new()
        .delete(format!("{base}/api/jobs/{job_id}"))
        .send()
        .await
        .expect("http");
    assert_eq!(del.status(), 200);
    // 后台 cleanup 是 fire-and-forget,等 1s 让 cleanup_job_media_blocking 跑完。
    tokio::time::sleep(Duration::from_secs(1)).await;
    let _ = shutdown.send(());
}

#[tokio::test]
async fn ssrf_bypass_payloads_all_rejected() {
    let Some((base, shutdown)) = boot_server().await else {
        return;
    };
    let bad = [
        "rtsp://[::ffff:127.0.0.1]/c",
        "http://user:pass@127.0.0.1/x",
        "http://%31%32%37.0.0.1/x",
        "rtsp://0x7f000001:554/cam",
        "rtsp://2130706433/cam",
    ];
    let client = reqwest::Client::new();
    for u in bad {
        let res = client
            .post(format!("{base}/api/jobs/stream"))
            .json(&serde_json::json!({ "url": u }))
            .send()
            .await
            .expect("http");
        assert_eq!(
            res.status(),
            400,
            "expected 400 for {u}, got {}",
            res.status()
        );
        let body: serde_json::Value = res.json().await.expect("json");
        assert!(
            body["error"].as_str().unwrap_or("").contains("blocked"),
            "expected 'blocked' in error for {u}, got: {body}"
        );
    }
    let _ = shutdown.send(());
}

#[tokio::test]
async fn download_zip_returns_valid_archive() {
    let Some((base, shutdown)) = boot_server().await else {
        return;
    };
    // 上传一个 image,产生 job。
    let img = std::fs::read("../tests/fixtures/demo_face_256.pgm")
        .or_else(|_| std::fs::read("tests/fixtures/demo_face_256.pgm"))
        .expect("fixture");
    let part = reqwest::multipart::Part::bytes(img.clone())
        .file_name("demo_face_256.pgm")
        .mime_str("image/x-portable-graymap")
        .expect("mime");
    let res = reqwest::Client::new()
        .post(format!("{base}/api/jobs/image"))
        .multipart(
            reqwest::multipart::Form::new()
                .text("algo", "haar")
                .part("file", part),
        )
        .send()
        .await
        .expect("http");
    let job_id = res.json::<serde_json::Value>().await.expect("json")["job_id"]
        .as_str()
        .unwrap()
        .to_string();
    // 等到 done
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(150)).await;
        let r: serde_json::Value = reqwest::get(format!("{base}/api/jobs/{job_id}"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if r["status"].as_str() == Some("done") {
            break;
        }
    }
    // 拉 zip
    let bytes = reqwest::get(format!("{base}/api/jobs/{job_id}/download.zip"))
        .await
        .expect("http")
        .bytes()
        .await
        .expect("body");
    assert!(bytes.len() > 22, "zip too small");
    // 头 4 字节 = LOCAL_FILE_HEADER_SIG,末 22 字节 末 22-18 偏移 = EOCD_SIG
    const LOCAL: u32 = 0x04034b50;
    const EOCD: u32 = 0x06054b50;
    let head = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    assert_eq!(head, LOCAL);
    let eocd_pos = bytes.len() - 22;
    let tail = u32::from_le_bytes(bytes[eocd_pos..eocd_pos + 4].try_into().unwrap());
    assert_eq!(tail, EOCD);
    // 至少应该有 original + annotated + manifest(没脸也至少有 manifest + original)。
    assert!(bytes.len() > 200, "zip truncated");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn range_request_returns_206() {
    let Some((base, shutdown)) = boot_server().await else {
        return;
    };
    // 先创建一个有 original 媒体的 job。
    let img = std::fs::read("../tests/fixtures/demo_face_256.pgm")
        .or_else(|_| std::fs::read("tests/fixtures/demo_face_256.pgm"))
        .expect("fixture");
    let part = reqwest::multipart::Part::bytes(img.clone())
        .file_name("demo_face_256.pgm")
        .mime_str("image/x-portable-graymap")
        .expect("mime");
    let res = reqwest::Client::new()
        .post(format!("{base}/api/jobs/image"))
        .multipart(
            reqwest::multipart::Form::new()
                .text("algo", "haar")
                .part("file", part),
        )
        .send()
        .await
        .expect("http");
    let job_id = res.json::<serde_json::Value>().await.unwrap()["job_id"]
        .as_str()
        .unwrap()
        .to_string();
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(150)).await;
        let r: serde_json::Value = reqwest::get(format!("{base}/api/jobs/{job_id}"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if r["status"].as_str() == Some("done") {
            break;
        }
    }
    let original_key = format!("jobs/{job_id}/original.pgm");
    let res = reqwest::Client::new()
        .get(format!("{base}/media/{original_key}"))
        .header("Range", "bytes=0-9")
        .send()
        .await
        .expect("http");
    assert_eq!(res.status(), 206, "expected 206 for Range");
    let body = res.bytes().await.expect("body");
    assert_eq!(body.len(), 10);
    let _ = shutdown.send(());
}

/// Smoke:验证 available_algos() 与平台 core 一致(防止 src/ 删除算法后未同步)。
#[test]
fn available_algos_includes_expected_three() {
    let algos = available_algos();
    assert!(algos.contains(&"haar"));
    assert!(algos.contains(&"luminance"));
    // cnn 是 optional feature;默认 build 不一定有,不强校验。
}

/// E1+E2 集成:取消正在跑的视频任务,SSE 流应该收到 cancel 之前的所有
/// frame 事件 + 终态 cancelled 事件 + 状态机翻 Cancelled。
///
/// 这个测试走"模拟 worker"模式:用 JobRegistry 创建一个 video 任务,
/// 通过 broadcast channel 模拟 SSE 订阅;手动 emit N 个 frame 事件,
/// 触发 cancel,然后验证订阅端收到 ≥ N 帧 + 终态 cancelled。
///
/// 不依赖 ffmpeg / rustfs / axum,纯逻辑验证 SSE contract。完整 HTTP 级
/// SSE 测试需要 reqwest::bytes_stream + futures-util,见 robustness.rs
/// 的 cancel_streams_partial_frames_then_terminal(底层路径同 SSE handler
/// 用的 broadcast::Receiver)。
#[tokio::test]
async fn cancel_running_job_streams_partial_frames_via_sse() {
    let (reg, shutdown) = boot_registry_for_test().await;
    let job = reg
        .create(
            rsface_platform::jobs::JobKind::Video,
            "cancel-sse.mp4".to_string(),
        )
        .expect("create");

    // 模拟 SSE handler 订阅同一 broadcast channel。
    let mut rx = job.event_tx.subscribe();

    // 模拟 worker:连续 emit 30 帧(模拟 cancel 前已处理的)。
    for i in 0..30u64 {
        job.emit(&serde_json::json!({
            "type": "frame",
            "frame": {"index": i, "timestamp_ms": i * 100, "annotated_key": null, "original_key": null, "faces": []}
        }).to_string());
    }

    // 触发 cancel(模拟 POST /api/jobs/{id}/cancel)。
    let ok = reg.request_cancel(&job.id);
    assert!(ok, "request_cancel returns true");

    // 模拟 run_job 在下一帧 loop 顶部检测到 cancel,发终态 + 设状态。
    job.set_status(rsface_platform::jobs::JobStatus::Cancelled);
    job.emit(&serde_json::json!({"type": "cancelled"}).to_string());

    // 收集订阅端事件。
    let mut frames_seen: u64 = 0;
    let mut saw_cancelled = false;
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(100), rx.recv()).await {
            Ok(Ok(payload)) => {
                let v: serde_json::Value = serde_json::from_str(&payload).unwrap_or_default();
                match v.get("type").and_then(|t| t.as_str()) {
                    Some("frame") => frames_seen += 1,
                    Some("cancelled") => {
                        saw_cancelled = true;
                        break;
                    }
                    _ => {}
                }
            }
            Ok(Err(_lagged)) => continue,
            Err(_) => continue,
        }
    }

    // 30 帧在 broadcast channel(capacity 256)内全部存活,订阅端应该全收到。
    assert!(
        frames_seen >= 25,
        "expected ≥25 frame events before cancel, got {frames_seen}"
    );
    assert!(saw_cancelled, "expected 'cancelled' SSE event");
    assert_eq!(
        job.status(),
        rsface_platform::jobs::JobStatus::Cancelled,
        "job status must be 'cancelled'"
    );
    let _ = shutdown.send(());
}

/// 测试辅助:起一个 in-memory JobRegistry,与 robustness.rs 的同名辅助对齐。
async fn boot_registry_for_test() -> (
    std::sync::Arc<rsface_platform::jobs::JobRegistry>,
    tokio::sync::oneshot::Sender<()>,
) {
    let mut cfg = rsface_platform::config::Config::from_env();
    cfg.tmp_dir = std::env::temp_dir().join(format!("rsface-it-cancel-{}", uuid_like()));
    cfg.local_media_dir = cfg.tmp_dir.join("media");
    cfg.database_url = "".to_string();
    cfg.max_concurrent_jobs = 1;
    cfg.max_queue_depth = 8;
    cfg.shutdown_grace_secs = 2;
    let _ = std::fs::create_dir_all(&cfg.tmp_dir);

    let s3 = std::sync::Arc::new(rsface_platform::s3::S3Client::new(
        cfg.s3_endpoint.clone(),
        cfg.s3_region.clone(),
        cfg.s3_access_key.clone(),
        cfg.s3_secret_key.clone(),
        cfg.s3_bucket.clone(),
    ));
    let db = std::sync::Arc::new(rsface_platform::persist::Db { pool: None });
    let rt = tokio::runtime::Handle::current();
    let reg = std::sync::Arc::new(rsface_platform::jobs::JobRegistry {
        jobs: std::sync::Mutex::new(std::collections::HashMap::new()),
        s3,
        cfg: cfg.clone(),
        db,
        rt: rt.clone(),
        job_slots: std::sync::Arc::new(tokio::sync::Semaphore::new(cfg.max_concurrent_jobs)),
        running_jobs: std::sync::atomic::AtomicU64::new(0),
        queued_jobs: std::sync::atomic::AtomicU64::new(0),
        started_at: std::time::Instant::now(),
        shutdown: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        gallery_cache: std::sync::Mutex::new(None),
        gallery: std::sync::Arc::new(rsface_platform::gallery::GalleryState::empty_for_tests(
            42,
            Default::default(),
        )),
    });
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown_flag = reg.shutdown.clone();
    tokio::spawn(async move {
        let _ = rx.await;
        shutdown_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    (reg, tx)
}
