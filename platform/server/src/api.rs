//! HTTP API + 静态前端托管 + S3 媒体代理。
//!
//! 路由:
//! - `GET  /`                     前端入口
//! - `GET  /{file}`               前端静态资源(web/)
//! - `GET  /api/health`           健康检查
//! - `GET  /api/config`           当前检测器模式(haar/cnn) + 权重文件状态
//! - `POST /api/jobs/image`       上传图片检测(multipart: file,≤ UPLOAD_LIMIT_IMAGE_MB)
//! - `POST /api/jobs/video`       上传视频检测(multipart: file,≤ UPLOAD_LIMIT_VIDEO_GB)
//! - `POST /api/jobs/stream`      直播流检测(JSON: {url})
//! - `GET  /api/jobs`             任务列表(summary;?limit=&offset= 分页)
//! - `GET  /api/jobs/stats`       按算法聚合成功/失败/平均耗时
//! - `GET  /api/jobs/{id}`        任务详情(帧 + 人脸)
//! - `POST /api/jobs/{id}/cancel` 取消任务
//! - `GET  /api/jobs/{id}/events` SSE 实时事件(直播流/进度)
//! - `POST|DELETE /api/jobs/batch` 批量操作(delete/archive/export)
//! - `GET  /api/metrics`          平台实时指标(前端 KPI 栏轮询,1s dedup 缓存)
//! - `GET  /media/{key}`          S3/本地 媒体代理(支持 Range/206)
//!
//! 性能层(2026-09-18 加):
//! - tower-http `CompressionLayer`:对 ≥1KB 响应自动 gzip(文本型 JSON 压缩率 70-90%)
//! - tower-http `SetResponseHeaderLayer`:静态资源 `Cache-Control: public, max-age=600, stale-while-revalidate=86400`,
//!   让前端 /style.css / /app.js 第二次加载走 304 + 缓存
//! - `/api/config`:启动期一次算好,后续直接 clone 缓存 bytes(避免每请求读 cfg + clone algo 数组)
//! - `/api/metrics`:1s TTL dedup,5 个 tab 同时打开也只算 1 次

use crate::cache::TtlCache;
use crate::jobs::{JobKind, JobRegistry, JobStatus};
use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tokio_util::io::ReaderStream;
use tower_http::compression::CompressionLayer;
use tower_http::set_header::SetResponseHeaderLayer;

/// 跨 handler 共享的 TTL 响应缓存。每条缓存可独立失效,粒度到端点。
#[derive(Clone)]
pub struct ResponseCaches {
    /// `/api/config` 启动期 immutable,TTL 1h。
    pub config_json: Arc<TtlCache>,
    /// `/api/metrics` 1s dedup,前端 KPI 轮询 50% 命中。
    pub metrics_json: Arc<TtlCache>,
    /// `/api/jobs`(列表)短 TTL dedup,前端 SSE 期间频繁轮询 50% 命中。
    pub jobs_list_json: Arc<TtlCache>,
    /// `/api/jobs/stats` 短 TTL,按算法聚合数据无状态机依赖,1s 复用足够。
    pub jobs_stats_json: Arc<TtlCache>,
}

pub fn router(state: Arc<JobRegistry>, caches: ResponseCaches) -> Router {
    // 上传上限分层:图片(默认 50MB)/ 视频(默认 2GB)。
    // 全局不再用 1GB 统一限制;其它路由(JSON/SSE)走 axum 默认 2MB。
    let img_limit = state.cfg.upload_limit_image;
    let video_limit = state.cfg.upload_limit_video;
    let cors_origin = state.cfg.cors_allow_origin.clone();
    let caches_for_routes = caches.clone();
    Router::new()
        .route("/", get(index))
        .route("/api/health", get(health))
        .route("/api/config", get(config_info))
        .route("/api/metrics", get(metrics))
        .route("/metrics", get(prometheus_metrics))
        .route("/api/jobs", get(list_jobs))
        .route("/api/jobs/stats", get(job_stats))
        .route("/api/jobs/batch", post(batch_ops).delete(batch_ops))
        .route(
            "/api/jobs/image",
            post(upload_image).layer(DefaultBodyLimit::max(img_limit)),
        )
        .route(
            "/api/jobs/video",
            post(upload_video).layer(DefaultBodyLimit::max(video_limit)),
        )
        .route("/api/jobs/stream", post(start_stream))
        .route("/api/jobs/{id}", get(job_detail).delete(delete_job))
        .route("/api/jobs/{id}/cancel", post(cancel_job))
        .route("/api/jobs/{id}/events", get(job_events))
        .route("/api/jobs/{id}/compare", post(compare_algos))
        .route("/api/jobs/{id}/download.zip", get(download_zip))
        .route("/api/jobs/{id}/retry", post(retry_job))
        // 导入端点 — 语义上"导入"一次性任务(视频文件 / 视频 URL),
        // 返回 job_id + S3 LAN URL,浏览器可直接播放。
        .route(
            "/api/import/video",
            post(import_video).layer(DefaultBodyLimit::max(video_limit)),
        )
        .route("/api/import/video-url", post(import_video_url))
        .route("/api/import/{id}/urls", get(import_urls))
        // 埋点:摄取 + 摘要(供前端 dashboard 轮询)
        .route("/api/telemetry", post(telemetry_ingest))
        .route("/api/telemetry/summary", get(telemetry_summary))
        .route("/api/telemetry/recent", get(telemetry_recent))
        .route("/media/{*key}", get(media))
        .route("/{file}", get(static_file))
        .layer(axum::middleware::from_fn(move |req, next| {
            cors_middleware(req, next, cors_origin.clone())
        }))
        // gzip 压缩层(>~1KB 自动启用,小响应直接走透传)。
        .layer(CompressionLayer::new())
        // 静态资源缓存头:web/* 1h+stale-while-revalidate 24h,
        // 让浏览器/CDN 把 /app.js / /style.css / 字体等强缓存。
        .layer(SetResponseHeaderLayer::if_not_present(
            axum::http::HeaderName::from_static("cache-control"),
            HeaderValue::from_static("public, max-age=600, stale-while-revalidate=86400"),
        ))
        .with_state((state, caches_for_routes))
}

/// 可选 CORS 中间件:`CORS_ALLOW_ORIGIN` 非空时启用(默认空 = 同源部署,
/// 完全不注入头,行为与旧版一致)。设置如 `CORS_ALLOW_ORIGIN=https://fe.example.com`
/// 后,所有响应带 `Access-Control-Allow-Origin`,OPTIONS 预检直接 204。
async fn cors_middleware(
    req: axum::extract::Request,
    next: Next,
    allow_origin: String,
) -> Response {
    if allow_origin.is_empty() {
        return next.run(req).await;
    }
    if req.method() == axum::http::Method::OPTIONS {
        let mut resp = StatusCode::NO_CONTENT.into_response();
        apply_cors_headers(resp.headers_mut(), &allow_origin);
        return resp;
    }
    let mut resp = next.run(req).await;
    apply_cors_headers(resp.headers_mut(), &allow_origin);
    resp
}

fn apply_cors_headers(headers: &mut axum::http::HeaderMap, allow_origin: &str) {
    let hv = |v: &str| {
        axum::http::HeaderValue::from_str(v).unwrap_or(axum::http::HeaderValue::from_static("*"))
    };
    headers.insert(
        axum::http::HeaderName::from_static("access-control-allow-origin"),
        hv(allow_origin),
    );
    headers.insert(
        axum::http::HeaderName::from_static("access-control-allow-methods"),
        hv("GET, POST, DELETE, OPTIONS"),
    );
    headers.insert(
        axum::http::HeaderName::from_static("access-control-allow-headers"),
        hv("Content-Type, Authorization, Range, Last-Event-ID"),
    );
    // telemetry ingest 走 sendBeacon 时,浏览器不触发预检,
    // 但跨域 fetch 仍可能触发 OPTIONS,所以保留常见 header 名。
    headers.insert(
        axum::http::HeaderName::from_static("access-control-max-age"),
        hv("86400"),
    );
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok", "service": "rsface-platform"}))
}

/// `GET /metrics` — Prometheus exposition format. Renders the same data as
/// `/api/metrics` (JSON) but in Prometheus 0.0.4 text format so a scraper can
/// ingest it without a JSON adapter. Always returns 200; failure modes are
/// impossible here because `to_prometheus` is allocation-only.
async fn prometheus_metrics(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
) -> (
    [(axum::http::HeaderName, axum::http::HeaderValue); 1],
    String,
) {
    let body = crate::metrics::render_prometheus(&state);
    let ct: axum::http::HeaderValue =
        axum::http::HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8");
    (
        [(axum::http::HeaderName::from_static("content-type"), ct)],
        body,
    )
}

/// 报告当前检测器模式 + 权重/级联文件状态 + 可用算法列表。
///
/// 模式选择规则(在 `jobs::build_detector` 中):
/// - `RSFACE_ALGO` 环境变量显式选择(`haar` / `cnn` / `luminance`)
/// - 否则:`use_cnn=true` 或 `cnn_weights` 路径已设置 → `"cnn"`
/// - 否则 → `"haar"`
///
/// 权重状态(仅 cnn 模式有意义):
/// - `template` — 未指定权重文件,使用 core 内置 hand-crafted 模板
/// - `available` — 路径已设置且文件存在
/// - `missing`  — 路径已设置但文件不存在(run_job 启动时会失败)
///
/// **性能**:启动后 immutable,1h TTL 缓存。命中路径 0 次 cfg 读取 + 0 次 algo clone。
async fn config_info(
    State((state, caches)): State<(Arc<JobRegistry>, ResponseCaches)>,
) -> Response {
    if let Some(bytes) = caches.config_json.get_fresh() {
        return cached_json_response(bytes);
    }
    let want_cnn = state.cfg.use_cnn || state.cfg.cnn_weights.is_some();
    let (mode, cnn_weights_path, cnn_weights_status) = if want_cnn {
        match &state.cfg.cnn_weights {
            Some(p) => {
                let status = if p.is_file() { "available" } else { "missing" };
                ("cnn", Some(p.display().to_string()), status.to_string())
            }
            None => ("cnn", None, "template".to_string()),
        }
    } else {
        ("haar", None, "n/a".to_string())
    };
    let cascade_status = if state.cfg.cascade_path.is_file() {
        "available"
    } else {
        "missing"
    };
    let body = serde_json::json!({
        "mode": mode,
        "algo": mode,
        "available_algos": crate::jobs::available_algos(),
        "cnn": {
            "weights_path": cnn_weights_path,
            "weights_status": cnn_weights_status,
            "use_cnn": state.cfg.use_cnn,
        },
        "haar": {
            "cascade_path": state.cfg.cascade_path.display().to_string(),
            "cascade_status": cascade_status,
        },
        "min_face_size": state.cfg.min_face_size,
    });
    let bytes = serde_json::to_vec(&body).unwrap_or_default();
    caches.config_json.put(bytes.clone());
    cached_json_response(bytes)
}

async fn index(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    headers: HeaderMap,
) -> Response {
    serve_static_with_304(
        &state.cfg.web_dir,
        "index.html",
        headers
            .get(header::IF_NONE_MATCH)
            .and_then(|v| v.to_str().ok()),
    )
    .await
}

async fn static_file(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Path(file): Path<String>,
    headers: HeaderMap,
) -> Response {
    serve_static_with_304(
        &state.cfg.web_dir,
        &file,
        headers
            .get(header::IF_NONE_MATCH)
            .and_then(|v| v.to_str().ok()),
    )
    .await
}

fn content_type_for(path: &str) -> &'static str {
    match path
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "html" => "text/html; charset=utf-8",
        "js" => "application/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        "json" => "application/json",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "pgm" => "image/x-portable-graymap",
        "ppm" => "image/x-portable-pixmap",
        _ => "application/octet-stream",
    }
}

async fn serve_static(web_dir: &std::path::Path, file: &str) -> Response {
    // 防目录穿越。
    if file.contains("..") || file.contains('\\') {
        return (StatusCode::BAD_REQUEST, "bad path").into_response();
    }
    let path = web_dir.join(file);
    // ETag:用文件名+长度+mtime 算 weak ETag,
    // 让浏览器 / CDN 在客户端 reload 时走 304(0 字节 body),
    // 不需要再重新传 .css/.js。
    let metadata = std::fs::metadata(&path).ok();
    let etag = metadata.as_ref().map(|m| {
        let len = m.len();
        let mtime = m
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        format!("W/\"rsface-{len:x}-{mtime:x}\"")
    });
    match tokio::fs::read(&path).await {
        Ok(bytes) => {
            let ct = content_type_for(file);
            let mut resp = ([(header::CONTENT_TYPE, ct)], bytes).into_response();
            if let Some(et) = etag {
                if let Ok(hv) = HeaderValue::from_str(&et) {
                    resp.headers_mut().insert(header::ETAG, hv);
                }
            }
            resp
        }
        Err(_) => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

/// 静态文件响应(支持 304 Not Modified):axum 默认不处理 `If-None-Match` / `If-Modified-Since`,
/// 我们自己比对客户端发来的 ETag,命中直接 304 + 0 字节 body。
/// 对一个 100KB 的 app.js,304 让后续 reload 的 body 节省 99.99% 网络。
async fn serve_static_with_304(
    web_dir: &std::path::Path,
    file: &str,
    if_none_match: Option<&str>,
) -> Response {
    let resp = serve_static(web_dir, file).await;
    let server_etag = resp
        .headers()
        .get(header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    if let (Some(s_etag), Some(c_etag)) = (server_etag.as_deref(), if_none_match) {
        // 比较:server ETag 是 W/"..." weak,客户端 If-None-Match 通常也是 W/"..."
        // 简单做法:直接 strip W/ 前后做字符串相等。
        if etag_eq(s_etag, c_etag) {
            return StatusCode::NOT_MODIFIED.into_response();
        }
    }
    resp
}

/// 弱 ETag 等价比较:`W/"a-b"` 与 `If-None-Match: W/"a-b", W/"c-d"`(多值逗号分隔)。
/// 我们只支持单值(常见用法)。
fn etag_eq(server: &str, client_hdr: &str) -> bool {
    fn strip(s: &str) -> &str {
        s.strip_prefix("W/").unwrap_or(s).trim()
    }
    let client_first = client_hdr.split(',').next().unwrap_or("").trim();
    strip(server) == strip(client_first)
}

/// 把 TTL 缓存命中字节原样返回(带 application/json content-type)。
fn cached_json_response(bytes: Vec<u8>) -> Response {
    ([(header::CONTENT_TYPE, "application/json")], bytes).into_response()
}

/// `GET /api/jobs` 查询参数。
/// 默认(无参数)行为与旧版完全一致:返回全部 job 的 summary(不含 frames)。
/// - `?limit=N&offset=M`:分页,返回 `{jobs, total, limit, offset}`;
/// - `?limit=N` 单独使用也返回 total 字段(方便前端显示总数)。
///
/// 注意:summary 本身就不含 frames 数组,全量帧数据走 `/api/jobs/{id}`。
#[derive(Default, Deserialize)]
struct ListJobsQuery {
    limit: Option<usize>,
    offset: Option<usize>,
}

async fn list_jobs(
    State((state, caches)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Query(q): Query<ListJobsQuery>,
    headers: HeaderMap,
) -> Response {
    // 缓存键:无参 = 完整列表;有 limit/offset = 不缓存(分页通常与视图绑定)。
    let cacheable = q.limit.is_none() && q.offset.is_none();
    if cacheable {
        if let Some(bytes) = caches.jobs_list_json.get_fresh() {
            // ETag 304 路径:客户端发 If-None-Match 且 hash 命中时,直接 304。
            let etag = format!("W/\"jobs-list-{:x}\"", fxhash_short(&bytes));
            if if_none_match_matches(
                headers
                    .get(header::IF_NONE_MATCH)
                    .and_then(|v| v.to_str().ok()),
                &etag,
            ) {
                return StatusCode::NOT_MODIFIED.into_response();
            }
            return cached_json_response_with_etag(bytes, &etag);
        }
    }
    let all = state.list();
    let total = all.len();
    // 无 limit/offset:保持旧版响应 shape(只有 jobs 数组,向后兼容)。
    let body = if cacheable {
        let jobs: Vec<serde_json::Value> = all.iter().map(|j| j.summary()).collect();
        serde_json::json!({"jobs": jobs})
    } else {
        let offset = q.offset.unwrap_or(0).min(total);
        let limit = q.limit.unwrap_or(total.saturating_sub(offset));
        let jobs: Vec<serde_json::Value> = all
            .iter()
            .skip(offset)
            .take(limit)
            .map(|j| j.summary())
            .collect();
        serde_json::json!({
            "jobs": jobs,
            "total": total,
            "limit": limit,
            "offset": offset,
        })
    };
    let bytes = serde_json::to_vec(&body).unwrap_or_default();
    if cacheable {
        caches.jobs_list_json.put(bytes.clone());
        let etag = format!("W/\"jobs-list-{:x}\"", fxhash_short(&bytes));
        cached_json_response_with_etag(bytes, &etag)
    } else {
        ([(header::CONTENT_TYPE, "application/json")], bytes).into_response()
    }
}

/// `GET /api/jobs/stats`:按算法(haar/cnn/luminance)聚合
/// 成功/失败/取消/平均耗时/检出数。queued 未定算法的归入 `pending`;
/// 历史数据中的已下线算法名(yunet/mtcnn/hog)按原始字符串原样分桶展示。
async fn job_stats(State((state, caches)): State<(Arc<JobRegistry>, ResponseCaches)>) -> Response {
    if let Some(bytes) = caches.jobs_stats_json.get_fresh() {
        return cached_json_response(bytes);
    }
    let samples = state.collect_agg_samples();
    let agg = crate::jobs::aggregate_algo_stats(&samples);
    let algos: Vec<serde_json::Value> = agg
        .iter()
        .map(|(name, a)| {
            serde_json::json!({
                "algo": name,
                "total": a.total,
                "done": a.done,
                "cancelled": a.cancelled,
                "error": a.error,
                "active": a.active,
                "avg_elapsed_ms": a.elapsed_ms_sum.checked_div(a.timed_count).unwrap_or(0),
                "total_elapsed_ms": a.elapsed_ms_sum,
                "detections": a.detections,
            })
        })
        .collect();
    let total_jobs: u64 = agg.values().map(|a| a.total + a.active).sum();
    let body = serde_json::json!({
        "algos": algos,
        "total_jobs": total_jobs,
    });
    let bytes = serde_json::to_vec(&body).unwrap_or_default();
    caches.jobs_stats_json.put(bytes.clone());
    cached_json_response(bytes)
}

/// `GET /api/metrics`:平台实时指标(前端 KPI 栏每 2s 轮询)。
/// 字段与 web/app.js kpi.refresh() 的读取保持兼容:
/// running / max_concurrency / total_frames_processed / total_frames_with_face /
/// total_detections / errored / cancelled / mode。
/// live_fps_max / gpu_pct / cascade_pass_rate 平台侧暂无数据源,返回 0
/// (前端对 0 有降级显示,不会报错)。
async fn metrics(State((state, caches)): State<(Arc<JobRegistry>, ResponseCaches)>) -> Response {
    // 1s TTL dedup 缓存:命中路径 0 次 registry 扫描 + 0 次 mutex 获取。
    if let Some(bytes) = caches.metrics_json.get_fresh() {
        return cached_json_response(bytes);
    }
    let jobs = state.list();
    let mut running = 0u64;
    let mut queued = 0u64;
    let mut errored = 0u64;
    let mut cancelled = 0u64;
    let mut frames_processed = 0u64;
    let mut frames_with_face = 0u64;
    let mut detections = 0u64;
    for j in &jobs {
        match j.status() {
            JobStatus::Running => running += 1,
            JobStatus::Queued => queued += 1,
            JobStatus::Error => errored += 1,
            JobStatus::Cancelled => cancelled += 1,
            JobStatus::Done => {}
        }
        // 每个 job 只短持一次 stats 锁,不在 await 点持有。poison 容忍:
        // 单个 job 线程 panic 过不应该拖垮聚合接口。
        let st = j.stats.lock().unwrap_or_else(|e| e.into_inner()).clone();
        frames_processed += st.frames_processed;
        frames_with_face += st.frames_with_face;
        detections += st.total_detections;
    }
    let mode = std::env::var("RSFACE_ALGO")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            if state.cfg.use_cnn || state.cfg.cnn_weights.is_some() {
                "cnn".into()
            } else {
                "haar".into()
            }
        });
    let body = serde_json::json!({
        "running": running,
        "queued": queued,
        "max_concurrency": state.cfg.max_concurrent_jobs,
        "total_frames_processed": frames_processed,
        "total_frames_with_face": frames_with_face,
        "total_detections": detections,
        "errored": errored,
        "cancelled": cancelled,
        "mode": mode,
        "live_fps_max": 0,
        "gpu_pct": 0,
        "total_gpu_levels": 0,
        "total_cpu_levels": 0,
        "cascade_pass_rate": 0.0,
    });
    let bytes = serde_json::to_vec(&body).unwrap_or_default();
    caches.metrics_json.put(bytes.clone());
    cached_json_response(bytes)
}

async fn job_detail(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Path(id): Path<String>,
) -> Response {
    match state.get(&id) {
        Some(job) => {
            let mut v = job.summary();
            v["frames"] =
                serde_json::to_value(&*job.frames.lock().unwrap_or_else(|e| e.into_inner()))
                    .unwrap_or_default();
            Json(v).into_response()
        }
        None => error_response(StatusCode::NOT_FOUND, "no such job"),
    }
}

async fn cancel_job(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Path(id): Path<String>,
) -> Response {
    match state.get(&id) {
        Some(job) => {
            job.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            Json(serde_json::json!({"ok": true})).into_response()
        }
        None => error_response(StatusCode::NOT_FOUND, "no such job"),
    }
}

async fn delete_job(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Path(id): Path<String>,
) -> Response {
    state.request_cancel(&id);
    let existed = state.remove(&id);
    let db = state.db.clone();
    let id_db = id.clone();
    tokio::spawn(async move {
        db.delete_job(&id_db).await;
    });
    // 删除孤儿媒体:S3 `jobs/{id}/` 前缀全部对象 + 本地 media/tmp 目录。
    state.spawn_media_cleanup(&id);
    if existed {
        Json(serde_json::json!({"ok": true, "deleted": id})).into_response()
    } else {
        Json(serde_json::json!({"ok": true, "deleted": id, "from_db_only": true})).into_response()
    }
}

#[derive(Deserialize)]
struct BatchReq {
    ids: Vec<String>,
    /// POST body 里的操作名;DELETE 方法或未传时默认 "delete"。
    op: Option<String>,
}

/// `POST /api/jobs/batch`(旧契约,body: {ids, op: delete|archive|export})与
/// `DELETE /api/jobs/batch`(新便捷端点,body: {ids},语义恒为 delete)共用。
async fn batch_ops(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Json(req): Json<BatchReq>,
) -> Response {
    if req.ids.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "ids must not be empty");
    }
    let op = req.op.clone().unwrap_or_else(|| "delete".to_string());
    match op.as_str() {
        "delete" => {
            for id in &req.ids {
                state.request_cancel(id);
            }
            let removed = state.remove_many(&req.ids);
            let db = state.db.clone();
            let ids = req.ids.clone();
            tokio::spawn(async move {
                db.delete_jobs(&ids).await;
            });
            // 同单任务删除:逐个清理各自的媒体前缀。
            for jid in &req.ids {
                state.spawn_media_cleanup(jid);
            }
            Json(serde_json::json!({"ok": true, "op": "delete", "requested": req.ids.len(), "removed_in_mem": removed.iter().filter(|x| **x).count()})).into_response()
        }
        "archive" => {
            let mut n = 0;
            for id in &req.ids {
                if state.set_archived(id, true) {
                    n += 1;
                }
            }
            Json(serde_json::json!({"ok": true, "op": "archive", "archived": n})).into_response()
        }
        "export" => {
            let mut jobs = Vec::new();
            for id in &req.ids {
                if let Some(j) = state.get(id) {
                    let mut s = j.summary();
                    let frames = j.frames.lock().unwrap_or_else(|e| e.into_inner());
                    s["frames"] = serde_json::to_value(&*frames).unwrap_or_default();
                    jobs.push(s);
                }
            }
            Json(serde_json::json!({"ok": true, "op": "export", "jobs": jobs})).into_response()
        }
        _ => error_response(
            StatusCode::BAD_REQUEST,
            "op must be one of: delete|archive|export",
        ),
    }
}

async fn retry_job(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Path(id): Path<String>,
) -> Response {
    let original = {
        let Some(j) = state.get(&id) else {
            return error_response(StatusCode::NOT_FOUND, "no such job");
        };
        let inp = j
            .original_input
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let k = j.kind;
        (inp, k)
    };
    let (inp, kind) = original;
    let Some(input) = inp else {
        return error_response(StatusCode::BAD_REQUEST, "job has no original_input");
    };
    if kind == JobKind::Image {
        return error_response(
            StatusCode::BAD_REQUEST,
            "image retry requires re-upload; use /api/jobs/image",
        );
    }
    let display = input.clone();
    let job = match state.create(kind, display) {
        Ok(j) => j,
        Err(e) => return error_response(StatusCode::TOO_MANY_REQUESTS, &e.to_string()),
    };
    let new_id = job.id.clone();
    state.set_original_input(&new_id, input.clone());
    state.spawn_run(job, input);
    Json(serde_json::json!({"ok": true, "job_id": new_id})).into_response()
}

#[derive(serde::Deserialize, Default)]
struct CompareQuery {
    /// Comma-separated algo list, e.g. `haar,cnn,luminance`. Optional —
    /// if missing, runs all 3 available algos.
    #[serde(default)]
    algos: Option<String>,
    /// Frame index for video/stream jobs (defaults to first frame with faces, or 0).
    /// 预留字段:当前实现固定用第一帧;保留解析以兼容已有调用方。
    #[serde(default)]
    #[allow(dead_code)]
    frame: Option<u64>,
}

/// `POST /api/jobs/{id}/compare?algos=haar,cnn,luminance`
/// 对任务的第一张图(或指定 frame)同时跑多个算法,返回每个算法的
/// detection 数、耗时、bounding boxes。前端用它来渲染 3 张并排小图
/// 的"算法对比"视图。未知 / 已下线算法名会被过滤,不参与构造。
async fn compare_algos(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Path(id): Path<String>,
    Query(q): Query<CompareQuery>,
) -> Response {
    // 1) 解析 algos 参数
    let requested: Vec<String> = match &q.algos {
        Some(s) if !s.is_empty() => s
            .split(',')
            .map(|x| x.trim().to_ascii_lowercase())
            .filter(|x| !x.is_empty())
            .collect(),
        _ => crate::jobs::available_algos()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    let valid: Vec<String> = requested
        .into_iter()
        .filter(|a| crate::jobs::available_algos().contains(&a.as_str()))
        .collect();
    if valid.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "no valid algos requested");
    }

    // 中 #8 + 安全 #2 + #3:compare_algos 之前 `fs::read` / `get_object` 把
    // 整段原视频灌进内存(2 GB+),对流任务没意义,而且一旦输入是 2 GB 视频
    // 就是 GB 级堆分配 → OOM。compare 实际只关心首帧,直接拒非图片 + 大于
    // 16 MiB 的输入,让前端切到 video-specific compare 路径(或先下载缩略图)。
    // S3 路径走 `get_object_range(0, +16 MiB)` 一次性读到上限,不存在
    // HEAD→GET TOCTOU;local 路径 stat 先判断。
    const COMPARE_MAX_BYTES: u64 = 16 * 1024 * 1024;

    // 2) 拿到 job 对应的原始媒体字节(S3 优先,失败回退到 local media dir)。
    let job = match state.get(&id) {
        Some(j) => j,
        None => return error_response(StatusCode::NOT_FOUND, "no such job"),
    };
    let media_key = job
        .original_media_key
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let media_key = match media_key {
        Some(k) => k,
        None => {
            return error_response(
                StatusCode::NOT_FOUND,
                "job has no original media (stream jobs not supported)",
            )
        }
    };
    // 中 #8:compare_algos 之前 `fs::read` / `get_object` 把整段原视频灌进
    // 内存(2 GB+),对流任务没意义,而且一旦输入是 2 GB 视频就是 GB 级
    // 堆分配 → OOM。compare 实际只关心首帧,直接拒非图片 + 大于 16 MiB 的
    // 输入,让前端切到 video-specific compare 路径(或先下载缩略图)。
    let ext_is_image = {
        let ext = std::path::Path::new(
            media_key
                .rsplit_once('/')
                .map(|(_, n)| n)
                .unwrap_or(&media_key),
        )
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
        matches!(
            ext.as_str(),
            "png" | "pgm" | "ppm" | "jpg" | "jpeg" | "bmp" | "webp"
        )
    };
    if !ext_is_image {
        return error_response(
            StatusCode::BAD_REQUEST,
            "compare_algos only supports image jobs (videos / streams require a separate per-frame endpoint)",
        );
    }
    let bytes: Vec<u8> = {
        if let Some(rest) = media_key.strip_prefix("local://") {
            let path = state.cfg.local_media_dir.join(rest);
            // 安全 #2(局部版):stat 拒大对象,跟 S3 路径对齐。比对 S3 路径
            // 简单是因为 tokio::fs 不必绕 spawn_blocking,元数据读天然短小。
            match tokio::fs::metadata(&path).await {
                Ok(m) if m.len() > COMPARE_MAX_BYTES => {
                    return error_response(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        &format!(
                            "original media too large for compare ({} > 16 MiB)",
                            m.len()
                        ),
                    );
                }
                Err(e) => {
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &format!("stat local media: {e}"),
                    );
                }
                _ => {}
            }
            match tokio::fs::read(&path).await {
                Ok(b) => b,
                Err(e) => {
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &format!("read local media: {e}"),
                    )
                }
            }
        } else if let Some(rest) = media_key.strip_prefix("s3://") {
            let s3 = state.s3.clone();
            let owned = rest.to_string();
            // 安全 #3(局部版):用 Range GET 取首 `COMPARE_MAX_BYTES + 1` 字节
            // 同时拿到完整数据流(若真超 16 MiB,中途长度超限直接拒)。
            // 这样 HEAD→GET TOCTOU 窗口不存在;GET 本身就 16 MiB+1 hard cap。
            let res = tokio::task::spawn_blocking(move || {
                s3.get_object_range(&owned, 0, Some(COMPARE_MAX_BYTES))
            })
            .await;
            match res {
                Ok(Ok((b, _total))) => {
                    if b.len() as u64 > COMPARE_MAX_BYTES {
                        return error_response(
                            StatusCode::PAYLOAD_TOO_LARGE,
                            "original media too large for compare",
                        );
                    }
                    b
                }
                _ => return error_response(StatusCode::NOT_FOUND, "S3 object not found"),
            }
        } else {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "media key has no scheme");
        }
    };

    // 3) 把媒体字节解成 GrayImage(只支持 PNG/PGM/PPM,JPG 走 ffmpeg 转 PGM)。
    let gray = match decode_to_gray(&bytes, &state.cfg.tmp_dir, &id).await {
        Ok(g) => g,
        Err(e) => {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("decode: {e}"))
        }
    };

    // 4) CPU 检测整段移入 spawn_blocking,避免在 tokio worker 上同步执行检测
    //    (CnnDetector 持有 !Sync scratch、DetectorKind !Sync,无法跨 async 任务
    //    共享,故在闭包内构造/串行执行/丢弃,而非并行)。
    let requested_algos = valid.clone();
    let detect = tokio::task::spawn_blocking(move || {
        let width = gray.width();
        let height = gray.height();
        let mut results: Vec<serde_json::Value> = Vec::new();
        for algo in &valid {
            match crate::jobs::build_detector_by_name(algo) {
                Ok(det) => {
                    let t0 = std::time::Instant::now();
                    let dets = det.detect(&gray);
                    let elapsed_ms = t0.elapsed().as_millis() as u64;
                    results.push(serde_json::json!({
                        "algo": algo,
                        "detection_count": dets.len(),
                        "elapsed_ms": elapsed_ms,
                        "detections": dets.iter().map(|d| serde_json::json!({
                            "x": d.x, "y": d.y, "w": d.w, "h": d.h, "score": d.score,
                        })).collect::<Vec<_>>(),
                    }));
                }
                Err(e) => {
                    results.push(serde_json::json!({"algo": algo, "error": e.to_string()}));
                }
            }
        }
        (width, height, results)
    })
    .await;
    let (width, height, results) = match detect {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("detect join: {e}"),
            )
        }
    };

    Json(serde_json::json!({
        "job_id": id,
        "width": width,
        "height": height,
        "requested_algos": requested_algos,
        "results": results,
    }))
    .into_response()
}

/// ffmpeg 单图转换的硬超时:损坏/截断的输入可能让 ffmpeg 长时间不退出,
/// 必须有上限,超时即 kill。
const FFMPEG_IMAGE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// 同步版 ffmpeg 调用的超时包装:在 spawn_blocking 里用 std::thread 启
/// ffmpeg,在超时窗口内等 join;超时则 std::process::exit(2) 暴力杀进程
/// (子进程继承 stdio 句柄,fork 杀子进程足够;若有 shell 包装再补一层
/// `pgrep ffmpeg | xargs kill` 兜底)。
///
/// 高 #6 / 中 #4:替换 upload_image + import_video_url 的 `Command::output()`
/// 裸调用;以前版本被损坏图片/慢直播源卡住,占死一个并发槽位。
fn run_ffmpeg_with_timeout_blocking(
    mut cmd: std::process::Command,
    timeout: std::time::Duration,
) -> std::io::Result<std::process::ExitStatus> {
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn()?;
    // 子进程 std::process::Child:用 busy-wait + try_wait 直到超时或退出。
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait(); // reap zombie
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("ffmpeg timeout after {timeout:?}"),
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => {
                let _ = child.kill();
                return Err(e);
            }
        }
    }
}

/// Decode arbitrary image bytes (PNG/PGM/PPM/JPG via ffmpeg) into a GrayImage.
/// Falls back to ffmpeg-based PGM conversion for JPG/WebP inputs that the
/// core codec doesn't recognise.
async fn decode_to_gray(
    bytes: &[u8],
    tmp_dir: &std::path::Path,
    job_id: &str,
) -> std::io::Result<rsface::image::GrayImage> {
    use rsface::image::png;
    // PNG: try the core decoder first.
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        let mut cur = std::io::Cursor::new(bytes);
        if let Ok(g) = png::decode_to_gray(&mut cur) {
            return Ok(g);
        }
    }
    // PGM: P5 binary.
    if bytes.starts_with(b"P5") {
        let mut cur = std::io::Cursor::new(bytes);
        return rsface::image::codec::read_pgm(&mut cur);
    }
    // PPM: P6 binary — convert to gray.
    if bytes.starts_with(b"P6") {
        let mut cur = std::io::Cursor::new(bytes);
        let rgb = rsface::image::codec::read_ppm(&mut cur)?;
        return Ok(rgb.to_gray());
    }
    // JPG / WebP / other: fall back to bounded ffmpeg → PGM.
    // 2026-09-20:旧实现用同步 Command::output()(无超时,损坏输入可让 ffmpeg
    // 永久挂起)+ std fs(阻塞 async runtime),且 compare 临时目录从不清理。
    // 改为 tokio::process + 硬超时 + kill,并在结束时删除临时目录。
    let work_dir = tmp_dir.join(format!("compare-{job_id}"));
    let conv: std::io::Result<rsface::image::GrayImage> = async {
        tokio::fs::create_dir_all(&work_dir).await?;
        let in_path = work_dir.join("input.bin");
        let out_path = work_dir.join("out.pgm");
        tokio::fs::write(&in_path, bytes).await?;
        let mut child = tokio::process::Command::new("ffmpeg")
            .args(["-y", "-i"])
            .arg(&in_path)
            .args(["-pix_fmt", "gray", "-f", "image2"])
            .arg(&out_path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| std::io::Error::other(format!("ffmpeg spawn: {e}")))?;
        // wait() 只借用 child,这样超时分支仍能 start_kill + 回收。
        let status = match tokio::time::timeout(FFMPEG_IMAGE_TIMEOUT, child.wait()).await {
            Ok(Ok(status)) => status,
            Ok(Err(e)) => return Err(std::io::Error::other(format!("ffmpeg wait: {e}"))),
            Err(_) => {
                // 超时:终止子进程并回收,避免泄漏。
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err(std::io::Error::other(format!(
                    "ffmpeg image convert timed out after {}s",
                    FFMPEG_IMAGE_TIMEOUT.as_secs()
                )));
            }
        };
        if !status.success() {
            // 单图转换 stderr 量很小,wait 之后再排空不会撑爆管道缓冲。
            let mut err = String::new();
            if let Some(mut se) = child.stderr.take() {
                use tokio::io::AsyncReadExt;
                let _ = se.read_to_string(&mut err).await;
            }
            return Err(std::io::Error::other(format!(
                "ffmpeg image convert failed: {err}"
            )));
        }
        let pgm = tokio::fs::read(&out_path).await?;
        tokio::task::spawn_blocking(move || {
            let mut cur = std::io::Cursor::new(pgm);
            rsface::image::codec::read_pgm(&mut cur)
        })
        .await
        .map_err(|e| std::io::Error::other(format!("pgm join: {e}")))?
    }
    .await;
    // 无论成功失败都清理 compare 临时目录。
    let _ = tokio::fs::remove_dir_all(&work_dir).await;
    conv
}

async fn upload_image(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    mp: Multipart,
) -> Response {
    handle_upload(state, mp, JobKind::Image).await
}

async fn upload_video(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    mp: Multipart,
) -> Response {
    handle_upload(state, mp, JobKind::Video).await
}

/// 上传字段分块落盘失败类型。
enum StagingError {
    /// 流式读取过程中累计字节已超过上限。
    TooLarge { len: u64 },
    /// 读字段或写暂存文件失败(客户端中断连接、磁盘错误等)。
    Io(std::io::Error),
}

/// 把 multipart `file` 字段分块直落到暂存文件,读取同时累加字节数;
/// 超过 `max_bytes` 立即中断并删掉暂存文件。旧实现 `field.bytes()` 把
/// GB 级视频整个读进内存(之后 `spawn_blocking` 还 clone 一份),并发
/// 上传极易 OOM。
async fn stream_field_to_staging(
    cfg: &crate::config::Config,
    mut field: axum::extract::multipart::Field<'_>,
    max_bytes: usize,
) -> Result<std::path::PathBuf, StagingError> {
    use tokio::io::AsyncWriteExt;

    let path = crate::jobs::staging_upload_path(cfg);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(StagingError::Io)?;
    }
    let mut file = tokio::fs::File::create(&path)
        .await
        .map_err(StagingError::Io)?;
    let mut total: u64 = 0;
    let outcome: Result<(), StagingError> = async {
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|e| StagingError::Io(std::io::Error::other(e.to_string())))?
        {
            total = total.saturating_add(chunk.len() as u64);
            if total > max_bytes as u64 {
                return Err(StagingError::TooLarge { len: total });
            }
            file.write_all(&chunk).await.map_err(StagingError::Io)?;
        }
        file.flush().await.map_err(StagingError::Io)?;
        Ok(())
    }
    .await;
    if let Err(e) = outcome {
        drop(file);
        let _ = tokio::fs::remove_file(&path).await;
        return Err(e);
    }
    Ok(path)
}

async fn handle_upload(state: Arc<JobRegistry>, mut mp: Multipart, kind: JobKind) -> Response {
    // 分层大小上限(路由层 DefaultBodyLimit 已按 kind 限制,这里再在
    // field 分块读取时做流式计数保护:超过即中断,不再把整个 body 缓进内存)。
    let max_bytes = match kind {
        JobKind::Image => state.cfg.upload_limit_image,
        _ => state.cfg.upload_limit_video,
    };
    // 取第一个 file 字段。
    // 同时提取 `algo` 字段(可选)作为本 job 的算法覆盖。
    let mut filename = None;
    let mut staged_path: Option<std::path::PathBuf> = None;
    let mut algo_override: Option<String> = None;
    while let Ok(Some(field)) = mp.next_field().await {
        match field.name() {
            Some("file") => {
                if staged_path.is_some() {
                    // 多个 file 字段:只接受第一个,后续直接忽略。
                    continue;
                }
                filename = field.file_name().map(|s| s.to_string());
                match stream_field_to_staging(&state.cfg, field, max_bytes).await {
                    Ok(p) => staged_path = Some(p),
                    Err(StagingError::TooLarge { len }) => {
                        return error_response(
                            StatusCode::PAYLOAD_TOO_LARGE,
                            &format!("upload too large: {} bytes (max {} bytes)", len, max_bytes),
                        );
                    }
                    Err(StagingError::Io(e)) => {
                        return error_response(
                            StatusCode::BAD_REQUEST,
                            &format!("read upload: {e}"),
                        )
                    }
                }
            }
            Some("algo") => {
                // 读取文本字段值;长度上限 32 防滥用。`set_algo_override`
                // 会再次校验是否属于 `available_algos()`。
                if let Ok(s) = field.text().await {
                    let trimmed = s.trim().to_string();
                    if !trimmed.is_empty() && trimmed.len() <= 32 {
                        algo_override = Some(trimmed);
                    }
                }
            }
            _ => {} // 忽略其它字段
        }
    }
    let Some(name) = filename else {
        if let Some(p) = staged_path {
            let _ = tokio::fs::remove_file(p).await;
        }
        return error_response(StatusCode::BAD_REQUEST, "missing 'file' field");
    };
    let Some(staged) = staged_path else {
        return error_response(StatusCode::BAD_REQUEST, "empty upload");
    };
    // 空文件(0 字节)直接拒绝并清理暂存。
    let is_empty = tokio::fs::metadata(&staged)
        .await
        .map(|m| m.len() == 0)
        .unwrap_or(true);
    if is_empty {
        let _ = tokio::fs::remove_file(&staged).await;
        return error_response(StatusCode::BAD_REQUEST, "empty upload");
    }

    let ext = sanitized_ext(&name, kind);
    let job = match state.create(kind, name.clone()) {
        Ok(j) => j,
        Err(e) => {
            let _ = tokio::fs::remove_file(&staged).await;
            return error_response(StatusCode::TOO_MANY_REQUESTS, &e.to_string());
        }
    };
    let id = job.id.clone();
    state.set_original_input(&id, name.clone());
    if let Some(a) = algo_override.take() {
        state.set_algo_override(&id, a);
    }

    // 暂存文件移入 job 工作目录(阻塞 IO 放 blocking 线程)。
    // Image kind:再对 PNG / JPG 用 ffmpeg 转 PPM,平台层兜底,core 的 PNG
    // 解码只支持 stored 块不再成为瓶颈;同时把用户的原始字节另行落到
    // local_media_dir/original.{ext},供 web 端的 preview 展示。
    let (path, pre_stored_original) = {
        let cfg = state.cfg.clone();
        let idc = id.clone();
        let kind_l = kind;
        let ext_l = ext.clone();
        let staged_l = staged.clone();
        match tokio::task::spawn_blocking(move || -> std::io::Result<(String, Option<String>)> {
            // 1) 暂存文件 rename 到 job 目录(给 ffmpeg / core 读)
            let raw_path = crate::jobs::move_staged_into_job(&cfg, &staged_l, &idc, &ext_l)?;
            // 2) Image:把 PNG/JPG 转成 PPM(RGB 三通道),core 直接吃
            //    旧实现是 PGM(灰度),导致标注/裁剪全是灰色。
            //    注意:这里用 ppm 还是走 core 的 read_ppm 路径,会同时拿到 gray + rgb。
            if kind_l == JobKind::Image
                && matches!(ext_l.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "webp")
            {
                let work = raw_path.parent().unwrap().to_path_buf();
                let ppm = work.join("input.ppm");
                let mut ffmpeg_cmd = std::process::Command::new("ffmpeg");
                ffmpeg_cmd
                    .args(["-y", "-loglevel", "error", "-i"])
                    .arg(&raw_path)
                    .args(["-pix_fmt", "rgb24", "-f", "image2"])
                    .arg(&ppm);
                let status = run_ffmpeg_with_timeout_blocking(ffmpeg_cmd, FFMPEG_IMAGE_TIMEOUT)
                    .map_err(|e| std::io::Error::other(format!("ffmpeg image→ppm: {e}")))?;
                if !status.success() || !ppm.is_file() {
                    return Err(std::io::Error::other("ffmpeg image→ppm failed"));
                }
                // 3) 用户的原始字节落到 local_media_dir,作 preview 用。
                //    图片上限小(MB 级),blocking 里读回内存可接受;视频路径
                //    全程不落内存,见 stream_field_to_staging。
                let bytes = std::fs::read(&raw_path)?;
                let display_key = format!("jobs/{idc}/original.{ext_l}");
                let ct = match ext_l.as_str() {
                    "png" => "image/png",
                    "jpg" | "jpeg" => "image/jpeg",
                    "bmp" => "image/bmp",
                    "webp" => "image/webp",
                    _ => "application/octet-stream",
                };
                let stored =
                    crate::jobs::put_bytes_with_fallback_blocking(&cfg, &display_key, ct, &bytes);
                Ok((ppm.to_string_lossy().to_string(), Some(stored)))
            } else {
                Ok((raw_path.to_string_lossy().to_string(), None))
            }
        })
        .await
        {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => {
                // 任务已 create() 占用了一个 queued 名额,但 prep 失败不会进入
                // run_job,必须回退计数并移除索引,否则累积到 max_queue_depth 后
                // 服务器会永久返回 429。同时清掉刚建的 job 工作目录/暂存残留。
                discard_created_job(&state, &id);
                remove_job_workdir(&state, &id);
                let _ = std::fs::remove_file(&staged);
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("upload prep: {e}"),
                );
            }
            Err(e) => {
                discard_created_job(&state, &id);
                remove_job_workdir(&state, &id);
                let _ = std::fs::remove_file(&staged);
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("upload join: {e}"),
                );
            }
        }
    };

    // 如果已经在 handle_upload 阶段存了 original,直接更新 job 的 original_media_key,
    // run_job 看到已设置就会跳过重复存储。
    if let Some(stored) = pre_stored_original {
        *job.original_media_key
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(stored.clone());
        let db = state.db.clone();
        let id_db = id.clone();
        let stored_db = stored.clone();
        tokio::spawn(async move {
            db.set_original_key(&id_db, &stored_db).await;
        });
    }
    let _ = name; // suppress unused warning in release

    state.spawn_run(job, path);
    Json(serde_json::json!({"job_id": id})).into_response()
}

#[derive(serde::Deserialize)]
struct StreamReq {
    url: String,
    /// 可选 per-job 算法覆盖(`haar` / `cnn` / ... / `luminance`),
    /// 无效值会被忽略并回退到 env/默认。
    algo: Option<String>,
}

async fn start_stream(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Json(req): Json<StreamReq>,
) -> Response {
    let url = req.url.trim().to_string();
    // 2026-09-20 security:允许 rtsp/http(s) 摄像头与 test:// 合成源,
    // 但拒绝 file:// —— 未认证调用方可借它让 ffmpeg 读容器内任意文件,
    // 再通过本 job 的 SSE 帧把内容取回(本地文件读取原语)。
    if url.len() > 2048
        || !(url.starts_with("rtsp://")
            || url.starts_with("http://")
            || url.starts_with("https://")
            || url.starts_with("test://"))
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            "url must be rtsp:// http(s):// or test://",
        );
    }
    // 解析一遍确认 host 非空(挡掉 `rtsp:///garbage` 这类畸形输入)。
    if !url.starts_with("test://") {
        if let Some(host) = url_authority_host(&url) {
            // 安全 #1+#2:`::ffff:127.0.0.1` / `%31%32%37.0.0.1` 在
            // `is_blocked_host` 之前先规范化;规范化后 reject 则 400。
            match normalize_host_for_check(host) {
                Some(norm) if is_blocked_host(norm.as_str()) => {
                    return error_response(
                        StatusCode::BAD_REQUEST,
                        "url host is blocked (loopback/link-local/private/metadata)",
                    );
                }
                None => {
                    return error_response(
                        StatusCode::BAD_REQUEST,
                        "url host contains forbidden characters",
                    );
                }
                _ => {}
            }
        } else {
            return error_response(StatusCode::BAD_REQUEST, "url has no host");
        }
    }
    let display = url.clone();
    let job = match state.create(JobKind::Stream, display) {
        Ok(j) => j,
        Err(e) => return error_response(StatusCode::TOO_MANY_REQUESTS, &e.to_string()),
    };
    let id = job.id.clone();
    state.set_original_input(&id, url.clone());
    if let Some(a) = req.algo {
        state.set_algo_override(&id, a);
    }
    state.spawn_run(job, url);
    Json(serde_json::json!({"job_id": id})).into_response()
}

/// 从 `scheme://[userinfo@]host[:port][/...]` 中取出 host(不含 userinfo/port)。
/// 不引入 url crate:这里只需确认 authority 的 host 部分非空。
/// IPv6 字面量必须按 URL 规范带方括号。
fn url_authority_host(url: &str) -> Option<&str> {
    let rest = url.split("://").nth(1)?;
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    // 去掉 userinfo(最后一个 @ 之后才是 host[:port])。
    let host_port = authority.rsplit('@').next()?;
    let host = if let Some(inside) = host_port.strip_prefix('[') {
        inside.split(']').next()?
    } else {
        // 去掉可选 :port;split 取第一个 ':' 之前;无 ':' 时返回整个串。
        host_port.split(':').next()?
    };
    if host.is_empty() {
        None
    } else {
        Some(host)
    }
}

/// 用户行为埋点接收端点。
///
/// 请求体:`{"events":[{"name":"page_view","ts":..,"props":{...},"session":..},...]}`
/// 一次性批量上报,减少 HTTP 开销;前端在 visibilitychange / beforeunload 时
/// 用 sendBeacon 兜底发出去。
///
/// **设计取舍**:
/// - 入库:若 db.pool 存在,写 telemetry 表(后置);不存在则只 stdout 打日志,
///   让本地无 DB 的开发模式也能跑。
/// - 失败隔离:解析失败 / 字段缺失 / 单条非法都不影响其它事件落库。
/// - PII 防御:服务端再脱一次密(服务端永远不信前端已脱过);
///   `props` 中的 `s3_key`、`/media/` URL、文件名、人脸 crop key 全部丢,
///   只保留聚合级指标(计数 / 维度)。
#[derive(Deserialize)]
struct TelemetryBatch {
    #[serde(default)]
    events: Vec<crate::persist::TelemetryEvent>,
}

async fn telemetry_ingest(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Json(body): Json<TelemetryBatch>,
) -> Response {
    let n = body.events.len();
    if n == 0 {
        return Json(serde_json::json!({"ok": true, "accepted": 0})).into_response();
    }
    if n > 500 {
        // 单批 500 上限,防恶意/异常写入压垮写入路径。
        return error_response(StatusCode::BAD_REQUEST, "batch too large (max 500 events)");
    }
    // 2026-09-20 security:只把通过服务端脱敏过滤的事件落库/计数。旧代码虽然
    // 计算了 accepted,却把未过滤的 `body.events.clone()` 写进 DB,使接口承诺
    // 的 PII 脱敏失效。这里过滤一次,落库与计数共用同一份。
    let safe: Vec<crate::persist::TelemetryEvent> =
        body.events.into_iter().filter(is_safe_event).collect();
    let accepted = safe.len();
    // 永远 stdout 一份,方便开发模式无 DB 也能看埋点。
    if let Some(first) = safe.first() {
        eprintln!(
            "[telemetry] batch {} events, accepted={} first.name={}",
            n, accepted, first.name
        );
    }
    // 有 DB → 异步批量落库
    if state.db.pool.is_some() && !safe.is_empty() {
        let db = state.db.clone();
        tokio::spawn(async move {
            db.insert_telemetry_batch(&safe).await;
        });
    }
    Json(serde_json::json!({"ok": true, "accepted": accepted})).into_response()
}

/// 服务端兜底的安全过滤:丢弃任何含敏感字段的事件(防御性,前端已经过滤过)。
fn is_safe_event(e: &crate::persist::TelemetryEvent) -> bool {
    if e.name.is_empty() || e.name.len() > 64 {
        return false;
    }
    // 服务端永远不信客户端的"已脱敏"声明;再扫一次敏感字段。
    // 低 #10:case-fold 比较,客户端换大小写/URL-encode 绕过全部拦下。
    let blob = format!("{:?}{:?}", e.name, e.props).to_ascii_lowercase();
    const FORBIDDEN: &[&str] = &[
        "s3://",
        "local://",
        "inline://", // 存储 key 前缀
        "/media/",   // 媒体代理路径
        "authorization",
        "bearer ",  // 鉴权相关
        "bearer\t", // 制表符分词也算
        "secret",   // 常见密钥字段
        "password",
        "token",
    ];
    !FORBIDDEN.iter().any(|s| blob.contains(s))
}

/// SSE 事件查询参数。`last_event_id` 是 SSE 协议约定的断点续传字段;
/// `?last_event_id=42` 表示客户端已经收到 id=42 之前的所有事件,只重发 >42 的。
#[derive(Default, Deserialize)]
struct EventsQuery {
    last_event_id: Option<u64>,
}

/// 并发 SSE 连接计数。每个连接 = 1 个 task + 256 深度 mpsc + 一个
/// broadcast 订阅,不加限制可被无限开连接耗尽内存。
static SSE_CONNECTIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// SSE 连接硬上限。LAN 单机产品,128 已远超合理的浏览器标签页数量。
const MAX_SSE_CONNECTIONS: u64 = 128;

/// 连接计数守卫:task 任何路径退出都会减计数(Drop 兜底)。
struct SseConnectionGuard;
impl Drop for SseConnectionGuard {
    fn drop(&mut self) {
        SSE_CONNECTIONS.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

async fn job_events(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Path(id): Path<String>,
    Query(q): Query<EventsQuery>,
) -> Response {
    let Some(job) = state.get(&id) else {
        return error_response(StatusCode::NOT_FOUND, "no such job");
    };
    // 连接数限制:超限直接拒绝,不再 subscribe/spawn。
    let active = SSE_CONNECTIONS.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    if active > MAX_SSE_CONNECTIONS {
        SSE_CONNECTIONS.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        return error_response(
            StatusCode::TOO_MANY_REQUESTS,
            "too many SSE connections, retry later",
        );
    }
    let rx = job.event_tx.subscribe();

    // 起始 event id = 客户端上次收到的;没传则从 0 开始。
    let start_after = q.last_event_id.unwrap_or(0);
    // 把 job 现有的帧当作历史回放,跳过 start_after 之前的。
    let mut seq: u64 = 0;
    let replay: Vec<(u64, String)> = {
        let frames = job.frames.lock().unwrap_or_else(|e| e.into_inner());
        frames
            .iter()
            .enumerate()
            .map(|(i, _fr)| {
                seq += 1;
                let payload = serde_json::json!({
                    "type": "replay",
                    "index": i,
                })
                .to_string();
                (seq, payload)
            })
            .collect()
    };

    let keepalive_secs = state.cfg.sse_keepalive_secs;
    // 停机标志:优雅停机时让空闲的 SSE 任务退出(否则 axum 的 graceful
    // shutdown 会一直等这些长连接,进程挂住不退)。
    let shutdown_flag = state.shutdown.clone();
    // 把所有事件(replay + live + keepalive)投到一个 tokio mpsc 通道,
    // 然后用 `ReceiverStream` 包成 Stream 给 axum 的 Sse。
    // 整个逻辑都在一个 task 里,handler 几乎瞬时返回 → 不占用 axum 工作线程。
    let (tx, rx_async) = tokio::sync::mpsc::channel::<Event>(256);
    tokio::spawn(async move {
        // 连接计数随 task 生命周期释放(任何退出/提前 return 都走 Drop)。
        let _connection_guard = SseConnectionGuard;
        // 1) 回放历史
        for (id, payload) in replay {
            if id <= start_after {
                continue;
            }
            if tx
                .send(Event::default().id(id.to_string()).data(payload))
                .await
                .is_err()
            {
                return; // 客户端已断开
            }
        }
        // 2) 订阅 live 事件 + 周期性 keepalive(keepalive=0 时也每 15s 醒来
        //    检查一次停机标志,避免 shutdown 时永久阻塞)。
        let mut live = BroadcastStream::new(rx);
        let mut seq_counter: u64 = seq;
        let poll_tick = if keepalive_secs > 0 {
            keepalive_secs
        } else {
            15
        };
        loop {
            let timeout = tokio::time::sleep(Duration::from_secs(poll_tick));
            tokio::pin!(timeout);
            let item = tokio::select! {
                item = live.next() => item,
                _ = &mut timeout => {
                    if keepalive_secs > 0
                        && tx.send(Event::default().comment("keepalive")).await.is_err()
                    {
                        return; // 客户端已断开
                    }
                    // 每次醒来检查停机:退出前给客户端发一个 done 信号,
                    // 前端可据此提示"服务重启中"。
                    if shutdown_flag.load(std::sync::atomic::Ordering::SeqCst) {
                        let _ = tx.send(Event::default()
                            .data(serde_json::json!({"type": "error", "message": "server shutting down"}).to_string()))
                            .await;
                        return;
                    }
                    continue;
                }
            };
            match item {
                Some(Ok(payload)) => {
                    // 类型化判断终态:旧实现用字符串子串匹配 JSON,字段顺序
                    // 或空格变化都会漏判;解析后取 type 字段与终态集合比较。
                    let done = sse_payload_is_terminal(&payload);
                    seq_counter += 1;
                    let evt = Event::default().id(seq_counter.to_string()).data(payload);
                    if tx.send(evt).await.is_err() {
                        return;
                    }
                    if done {
                        return;
                    }
                }
                Some(Err(_lagged)) => continue, // 慢消费者,继续等
                None => return,
            }
        }
    });
    // ReceiverStream 已经是 Stream<Item=Event>;wrap 成 Result<Event, Infallible>。
    // 不再叠加 axum KeepAlive:task 内部已按 sse_keepalive_secs 显式发
    // keepalive 注释帧(并顺带做停机检测),两层 keepalive 会重复发帧。
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx_async).map(Ok::<Event, Infallible>);
    Sse::new(stream).into_response()
}

/// 解析事件 JSON,type ∈ done/error/cancelled 时为终态(发送端任务
/// 随即关闭 SSE 连接,前端 EventSource 触发结束/错误回调)。
fn sse_payload_is_terminal(payload: &str) -> bool {
    const TERMINAL: &[&str] = &["done", "error", "cancelled"];
    serde_json::from_str::<serde_json::Value>(payload)
        .ok()
        .and_then(|v| v.get("type").and_then(|t| t.as_str()).map(str::to_string))
        .is_some_and(|t| TERMINAL.contains(&t.as_str()))
}

/// 解析 `Range: bytes=start-end` 请求头(仅支持单区间;多区间降级为 200 全量)。
/// 返回 Some((start, end_inclusive));end 为 None 表示到 EOF。
/// 不合规范(Satisfiable=false,start 超过文件长度)由调用方处理。
fn parse_range_header(hv: &str, total_len: u64) -> Option<(u64, Option<u64>)> {
    let spec = hv.strip_prefix("bytes=")?;
    // 多区间(逗号)不支持,直接走全量。
    if spec.contains(',') {
        return None;
    }
    let (s, e) = spec.split_once('-')?;
    let start: u64 = if s.is_empty() {
        // suffix range: `bytes=-N` 取最后 N 字节
        let n: u64 = e.parse().ok()?;
        return Some((
            total_len.saturating_sub(n),
            Some(total_len.saturating_sub(1)),
        ));
    } else {
        s.parse().ok()?
    };
    if start >= total_len {
        return Some((u64::MAX, None));
    } // unsatisfiable 标记
    let end = if e.is_empty() {
        None
    } else {
        e.parse::<u64>().ok().map(|v| v.min(total_len - 1))
    };
    Some((start, end))
}

/// 高 #3:本地媒体 Range 切片(流式)。返回 `tokio::fs::File` 加 seek 偏移,
/// 调用方用 `ReaderStream` 转 axum `Body::from_stream` 边读边发,避免
/// `read_exact` 把整段塞进 Vec(GB 级视频拖进度条的关键)。
async fn read_local_range_stream(
    path: &std::path::Path,
    start: u64,
    end_inclusive: Option<u64>,
) -> std::io::Result<(tokio::io::Take<tokio::fs::File>, u64)> {
    use tokio::io::{AsyncReadExt, AsyncSeekExt};
    let mut f = tokio::fs::File::open(path).await?;
    let total = f.metadata().await?.len();
    if start >= total {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "range start beyond EOF",
        ));
    }
    let end_inclusive = end_inclusive.unwrap_or(total - 1).min(total - 1);
    let len = end_inclusive - start + 1;
    f.seek(std::io::SeekFrom::Start(start)).await?;
    let limited = f.take(len);
    Ok((limited, total))
}

/// 高 #3:206 Partial Content 流式响应。从 AsyncRead 构造 axum Body,
/// 配上 Content-Range / Content-Length 头。注意 Content-Length 是
/// 服务端给的对象总大小(用于客户端下一段 Range 请求),body 实际只
/// 发对应片段。
fn stream_range_response(
    status: StatusCode,
    content_type: &'static str,
    cache_control: &'static str,
    content_range: &str,
    reader: impl tokio::io::AsyncRead + Send + 'static,
    _total: u64,
) -> Response {
    use axum::body::Body;
    let stream = ReaderStream::new(reader);
    let body = Body::from_stream(stream);
    let mut resp = (status, body).into_response();
    resp.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    resp.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(cache_control),
    );
    resp.headers_mut()
        .insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    resp.headers_mut().insert(
        header::CONTENT_RANGE,
        HeaderValue::from_str(content_range)
            .unwrap_or_else(|_| HeaderValue::from_static("bytes */*")),
    );
    resp
}

async fn media(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Path(key): Path<String>,
    headers: HeaderMap,
) -> Response {
    // key 可能由浏览器以两种形式送来:
    //   1) 原始:`local://jobs/.../original.mp4` 或 `s3://...` —— 由 put_with_fallback 写入
    //   2) 客户端错误地双重编码:`local%3A%2F%2Fjobs%2F...`(axum 已经把 % 解码为 :,所以两种都到这一步)
    //   3) 没有 scheme,直接是 key(老式调用残留)
    //
    // 处理:优先按 scheme 拆;剥掉 scheme 后是相对路径,然后:
    //   - `local` → 读 `state.cfg.local_media_dir/<real_key>`
    //   - `s3`    → 调 S3 client
    //   - 其余    → 视作 local 路径(S3 不可用时的降级模式,不再抛错)
    //
    // 容错:再保险地把 `local://` / `s3://` 再剥一次(防御性,即便 axum
    // 给我们的是已经 strip 过的 key 也不会出错)。
    let cleaned = key
        .strip_prefix("local://")
        .map(|s| s.to_string())
        .or_else(|| key.strip_prefix("s3://").map(|s| s.to_string()))
        .unwrap_or_else(|| key.clone());
    if cleaned.contains("..") || cleaned.contains('\\') {
        return error_response(StatusCode::BAD_REQUEST, "bad key");
    }
    // 2026-09-20 security:拒绝绝对路径。`Path::join` 遇到绝对路径的右值会
    // 直接丢弃 base,否则 `/media//etc/passwd`(或 URL 编码的 %2F)会解析到
    // media 根之外,造成未授权任意文件读取。
    if cleaned.starts_with('/') {
        return error_response(StatusCode::BAD_REQUEST, "bad key");
    }

    // `inline://` 兜底:这种 key 表示数据 base64 嵌在 SSE 事件的 `inline`
    // 字段里(见 jobs.rs 的 `put_with_inline_fallback`),前端必须通过 SSE
    // 拿到 base64 然后拼 `data:` URL,而不是走 /media/。这里返回 410 + 明确
    // 提示,避免前端误用 `/media/inline%3A%2F%2F...` 拿到空 404 后不知道
    // 是配置问题还是数据问题。
    if cleaned.starts_with("inline://") || key.starts_with("inline://") {
        return error_response(
            StatusCode::GONE,
            "inline:// keys must be fetched via SSE replay (look for `inline` field in frame events)",
        );
    }

    // Range 支持(视频拖动进度条):浏览器 / <video> 控件会发
    // `Range: bytes=N-M`。命中时回 206 + Content-Range;不支持/无头时
    // 维持旧版 200 全量行为。
    let range_hdr = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let ct = content_type_for(&cleaned);
    let cache = "public, max-age=3600";
    let local_path = state.cfg.local_media_dir.join(&cleaned);

    // ---- local 命中 ----
    if local_path.is_file() {
        // 2026-09-20 security:canonicalize 解析符号链接后,确认最终路径仍落在
        // media 根内,堵住"media 根内的 symlink 指向外部文件"这类逃逸。
        // is_file() 为真意味着根目录必然存在,两个 canonicalize 都应成功。
        let inside = match (
            tokio::fs::canonicalize(&state.cfg.local_media_dir).await,
            tokio::fs::canonicalize(&local_path).await,
        ) {
            (Ok(root), Ok(c)) => c.starts_with(root),
            _ => false,
        };
        if !inside {
            return error_response(StatusCode::BAD_REQUEST, "bad key");
        }
        // 先拿文件长度(无 Range 时也直接读,旧路径)。
        let total = match tokio::fs::metadata(&local_path).await {
            Ok(m) => m.len(),
            Err(_) => 0,
        };
        if let (Some(rh), true) = (&range_hdr, total > 0) {
            match parse_range_header(rh, total) {
                Some((u64::MAX, _)) => {
                    // unsatisfiable:416 + Content-Range: bytes */total
                    return (
                        StatusCode::RANGE_NOT_SATISFIABLE,
                        format!("bytes */{total}"),
                    )
                        .into_response();
                }
                Some((start, end)) => {
                    // 高 #3:用 seek + ReaderStream 流式切片,而不是 read_exact
                    // 把整段读到 Vec。GB 级视频拖进度条时峰值从 GB 降到 KB。
                    match read_local_range_stream(&local_path, start, end).await {
                        Ok((async_reader, total)) => {
                            let last = end.unwrap_or(total - 1);
                            return stream_range_response(
                                StatusCode::PARTIAL_CONTENT,
                                ct,
                                cache,
                                &format!("bytes {start}-{last}/{total}"),
                                async_reader,
                                total,
                            );
                        }
                        Err(_) => {
                            // 切片失败(文件刚好被删/截断):降级 200 全量重读。
                        }
                    }
                }
                None => { /* 多区间等不支持 → 走 200 全量 */ }
            }
        }
        return match tokio::fs::read(&local_path).await {
            Ok(bytes) => (
                [
                    (header::CONTENT_TYPE, ct),
                    (header::CACHE_CONTROL, cache),
                    (header::ACCEPT_RANGES, "bytes"),
                ],
                bytes,
            )
                .into_response(),
            Err(_) => error_response(StatusCode::NOT_FOUND, "local object not found"),
        };
    }

    // ---- local 不命中,降级到 S3 ----
    let s3 = state.s3.clone();
    let owned = cleaned.clone();
    // Range 路径需要先 HEAD 拿总长度;失败则退化为全量 GET。
    if let Some(rh) = &range_hdr {
        let s3h = s3.clone();
        let key_h = owned.clone();
        let head = tokio::task::spawn_blocking(move || s3h.head_object(&key_h)).await;
        if let Ok(Ok(total)) = head {
            match parse_range_header(rh, total) {
                Some((u64::MAX, _)) => {
                    return (
                        StatusCode::RANGE_NOT_SATISFIABLE,
                        format!("bytes */{total}"),
                    )
                        .into_response();
                }
                Some((start, end)) => {
                    let s3r = s3.clone();
                    let key_r = owned.clone();
                    let res = tokio::task::spawn_blocking(move || {
                        s3r.get_object_range_stream(&key_r, start, end)
                    })
                    .await;
                    if let Ok(Ok((stream, total))) = res {
                        let last = end.unwrap_or(total - 1);
                        return stream_range_response(
                            StatusCode::PARTIAL_CONTENT,
                            ct,
                            cache,
                            &format!("bytes {start}-{last}/{total}"),
                            stream,
                            total,
                        );
                    }
                    // Range GET 失败(可能 rustfs 不支持):降级全量 GET(下方)。
                }
                None => { /* 多区间 → 全量 */ }
            }
        }
    }
    let result = tokio::task::spawn_blocking(move || s3.get_object(&owned)).await;
    match result {
        Ok(Ok((bytes, _ct))) => {
            // 不信 S3 返回的 Content-Type(rustfs 经常给 octet-stream),
            // 用扩展名自己算,确保 <video>/<img> 能解码。
            (
                [
                    (header::CONTENT_TYPE, ct),
                    (header::CACHE_CONTROL, cache),
                    (header::ACCEPT_RANGES, "bytes"),
                ],
                bytes,
            )
                .into_response()
        }
        _ => {
            eprintln!(
                "[media] not found local='{}' and S3 lookup failed",
                local_path.display()
            );
            error_response(StatusCode::NOT_FOUND, "object not found")
        }
    }
}

fn sanitized_ext(name: &str, kind: JobKind) -> String {
    let ext = name
        .rsplit('.')
        .next()
        .unwrap_or("")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(5)
        .collect::<String>()
        .to_ascii_lowercase();
    if ext.is_empty() {
        match kind {
            JobKind::Image => "png".to_string(),
            _ => "mp4".to_string(),
        }
    } else {
        ext
    }
}

/// 按 media key(`local://` / `s3://`,或无 scheme 的 local 相对路径)读取整个
/// 对象。供 zip 导出收集各部分。
async fn read_media_object(state: &JobRegistry, key: &str) -> std::io::Result<Vec<u8>> {
    if let Some(rest) = key.strip_prefix("local://") {
        return tokio::fs::read(state.cfg.local_media_dir.join(rest)).await;
    }
    if let Some(rest) = key.strip_prefix("s3://") {
        let s3 = state.s3.clone();
        let k = rest.to_string();
        return match tokio::task::spawn_blocking(move || s3.get_object(&k)).await {
            Ok(Ok((b, _ct))) => Ok(b),
            Ok(Err(e)) => Err(std::io::Error::other(format!("s3 get: {e}"))),
            Err(e) => Err(std::io::Error::other(format!("s3 join: {e}"))),
        };
    }
    tokio::fs::read(state.cfg.local_media_dir.join(key)).await
}

/// 取 media key 末段的扩展名(最后一个点之后),校验为短的字母数字,异常时退回 bin。
fn key_extension(key: &str) -> String {
    let base = key.rsplit('/').next().unwrap_or(key);
    match base.rsplit('.').next() {
        Some(ext)
            if !ext.is_empty()
                && ext.len() <= 5
                && ext.chars().all(|c| c.is_ascii_alphanumeric()) =>
        {
            ext.to_ascii_lowercase()
        }
        _ => "bin".to_string(),
    }
}

/// `GET /api/jobs/{id}/download.zip` — 打包导出原始媒体 + 标注帧 + 人脸裁剪
/// + `manifest.json`。STORE 零压缩、零新增依赖(见 `zip.rs`)。
async fn download_zip(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Path(id): Path<String>,
) -> Response {
    let Some(job) = state.get(&id) else {
        return error_response(StatusCode::NOT_FOUND, "no such job");
    };
    // Mutex 中毒也不 panic:取内部值继续,避免单个任务毒锁把请求线程带挂。
    let frames = job.frames.lock().unwrap_or_else(|p| p.into_inner()).clone();
    let original_media = job
        .original_media_key
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    let algo = job.algo.lock().unwrap_or_else(|p| p.into_inner()).clone();
    let stats = job.stats.lock().unwrap_or_else(|p| p.into_inner()).clone();

    let mut zip = crate::zip::ZipWriter::new();
    let mut manifest_frames: Vec<serde_json::Value> = Vec::new();

    // 中 #9:download_zip 之前 `zip.finish()` 把全部 entry 一次进 Vec;视频 job
    // 几百帧 + 几千裁剪会到 GB 级。加全局字节上限,超过就跳过剩余 entry 并
    // 在 manifest 里写明,前端可看到 truncated。
    const ZIP_MAX_BYTES: usize = 256 * 1024 * 1024; // 256 MiB 软上限
    let mut total_written: usize = 0;
    let mut skipped_after_limit: usize = 0;
    let push_entry = |zip: &mut crate::zip::ZipWriter,
                      name: &str,
                      bytes: &[u8],
                      total_written: &mut usize,
                      skipped: &mut usize|
     -> bool {
        // ZipWriter 内部 guard:entry 超 4 GiB 拒收,这里再前置一道字节预算。
        let entry_size = bytes.len();
        if *total_written + entry_size > ZIP_MAX_BYTES {
            *skipped += 1;
            return false;
        }
        if zip.add_file(name, bytes).is_err() {
            return false;
        }
        *total_written += entry_size;
        true
    };

    // 原始媒体。
    if let Some(key) = &original_media {
        if let Ok(bytes) = read_media_object(&state, key).await {
            let ext = key_extension(key);
            let _ = push_entry(
                &mut zip,
                &format!("original/original.{ext}"),
                &bytes,
                &mut total_written,
                &mut skipped_after_limit,
            );
        }
    }

    // 标注帧 + 人脸裁剪。
    for fr in &frames {
        if skipped_after_limit > 0 {
            break;
        }
        let Some(ann_key) = &fr.annotated_key else {
            continue;
        };
        let Ok(ann_bytes) = read_media_object(&state, ann_key).await else {
            continue;
        };
        let frame_name = format!("annotated/frame_{:06}.png", fr.index);
        if !push_entry(
            &mut zip,
            &frame_name,
            &ann_bytes,
            &mut total_written,
            &mut skipped_after_limit,
        ) {
            continue;
        }
        let mut face_files: Vec<serde_json::Value> = Vec::new();
        for (i, f) in fr.faces.iter().enumerate() {
            if skipped_after_limit > 0 {
                break;
            }
            let Ok(crop) = read_media_object(&state, &f.key).await else {
                continue;
            };
            let face_name = format!("faces/frame_{:06}_face_{:02}.png", fr.index, i);
            if push_entry(
                &mut zip,
                &face_name,
                &crop,
                &mut total_written,
                &mut skipped_after_limit,
            ) {
                face_files.push(serde_json::json!({
                    "file": face_name,
                    "x": f.x, "y": f.y, "w": f.w, "h": f.h, "score": f.score,
                }));
            }
        }
        manifest_frames.push(serde_json::json!({
            "index": fr.index,
            "timestamp_ms": fr.timestamp_ms,
            "file": frame_name,
            "faces": face_files,
        }));
    }

    let exported_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let manifest = serde_json::json!({
        "job_id": job.id,
        "display_name": job.display_name,
        "kind": match job.kind {
            JobKind::Image => "image", JobKind::Video => "video", JobKind::Stream => "stream",
        },
        "algo": algo,
        "stats": stats,
        "frames": manifest_frames,
        "exported_ms": exported_ms,
        // 中 #9:字节超限时被跳过的 entry 数,前端可见。
        "truncated_entries": skipped_after_limit,
        "size_bytes": total_written,
    });
    let _ = zip.add_file("manifest.json", manifest.to_string().as_bytes());

    let bytes = zip.finish();
    // 文件名只用 id(hex-dash 安全字符),不嵌入可能含引号的 display_name。
    let id_head: String = id.chars().take(8).collect();
    let disposition = format!("attachment; filename=\"rsface-{id_head}.zip\"");
    (
        [
            (header::CONTENT_TYPE, "application/zip".to_string()),
            (header::CONTENT_DISPOSITION, disposition),
        ],
        bytes,
    )
        .into_response()
}

fn error_response(code: StatusCode, msg: &str) -> Response {
    (code, Json(serde_json::json!({"error": msg}))).into_response()
}

/// `JobRegistry::create()` 已成功、但任务在进入 `run_job` 之前就失败时的统一
/// 回滚:释放占用的 queued 计数 + 从内存索引移除 + 尽力删除 DB 里的 Queued 行,
/// 避免僵尸任务把队列计数顶到上限导致永久 429,以及 PG 里残留 Queued 行。
///
/// 注意:`create()` 内部以 fire-and-forget 方式异步 insert_job,与此处的 delete
/// 存在极小的竞态窗口(delete 先于 insert 提交则行会留下)。当前重启不做
/// hydration,残留 Queued 行无运行时影响;未来接入 hydration 时应让 insert
/// 以登记时的真实状态为准,届时彻底消除该竞态。
fn discard_created_job(state: &JobRegistry, id: &str) {
    state.abandon_job(id);
    let db = state.db.clone();
    let jid = id.to_string();
    tokio::spawn(async move {
        db.delete_job(&jid).await;
    });
}

/// 尽力删除 job 在 tmp_dir 下的工作目录(prep 阶段失败时的残留)。
/// 阻塞 remove_dir_all 放到 blocking 线程,失败只告警不阻断响应。
fn remove_job_workdir(state: &JobRegistry, id: &str) {
    let dir = state.cfg.tmp_dir.join(id);
    tokio::task::spawn_blocking(move || {
        if let Err(e) = std::fs::remove_dir_all(&dir) {
            if e.kind() != std::io::ErrorKind::NotFound {
                eprintln!("[jobs] cleanup workdir {dir:?} failed: {e}");
            }
        }
    });
}

// ============================================================================
// 视频导入端点 (2026-09-18 加)
//
// 设计目标:
// - `/api/import/video` (multipart) — 浏览器直接上传的视频文件,处理完返回
//   S3 LAN URL,浏览器用 <video> 直接播放;
// - `/api/import/video-url` (JSON {url, algo?}) — 服务端用 ffmpeg 把远程
//   http(s) 视频拉下来,走和上传视频一样的流水线(转 mp4 → 落 S3 → 跑检测);
// - `/api/import/{id}/urls` — 给前端"导入完成后"轮询 S3 URL 用的轻量端点,
//   不重发整份 job summary,省带宽。
//
// 关键差异 vs /api/jobs/{video,stream}:
// - 这两个是"导入"语义(用户预期处理完后能拿到播放链接),
//   不是"长跑"语义(流任务永远不会 done);
// - 因此新端点全部用 JobKind::Video,ffmpeg 一次性拉到本地,然后走完整的
//   run_job 流水线(原文件 + 标注帧 + 裁脸都会写到 S3)。
//
// LAN 播放:无论 S3 还是 local,前端都走 `/media/{key}` 路由(已支持 Range/206),
// 浏览器在同网段访问这台机器的 20080 即可直接播 mp4。
// ============================================================================

/// 包装一份"导入完成"响应:job_id + 一组 LAN URL + 当前状态。
fn import_response(
    id: &str,
    kind: JobKind,
    original_key: Option<&str>,
    cover_key: Option<&str>,
    status: JobStatus,
) -> Response {
    Json(serde_json::json!({
        "ok": true,
        "job_id": id,
        "kind": kind,
        "status": status,
        "original_url": media_lan_url(original_key),
        "cover_url": media_lan_url(cover_key),
        // 同一 key 的不同代理形式(让前端可任选):
        // - `media_path`:直接给 `/media/<encoded key>` 让 <video src> 用
        // - `stream_url`:SSE 事件流(`/api/jobs/{id}/events`)
        "media_path": media_lan_path(original_key),
        "stream_url": format!("/api/jobs/{}/events", id),
    }))
    .into_response()
}

/// 把 `s3://...` / `local://...` / `inline://...` 转成 LAN 代理路径。
/// 直接给前端 `<video src>` / `<a href>` 用;浏览器在同网段访问机器即可。
fn media_lan_url(key: Option<&str>) -> Option<String> {
    media_lan_path(key).map(|p| format!("/media/{p}"))
}

fn media_lan_path(key: Option<&str>) -> Option<String> {
    let k = key?;
    let stripped = k
        .strip_prefix("s3://")
        .or_else(|| k.strip_prefix("local://"))
        .or_else(|| k.strip_prefix("inline://"))
        .unwrap_or(k);
    // 浏览器在取 URL 时会再做一次 percent-decode,这里只 encode 路径段。
    Some(url_encode_path(stripped))
}

/// 极简 path 编码(只编 `?#& %` 等保留字符;保留 `/` `-` `_` `.`)。
/// 0-dep,避免引入 urlencoding crate。
fn url_encode_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for b in s.bytes() {
        let safe = matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9'
            | b'-' | b'_' | b'.' | b'~' | b'/' | b':');
        if safe {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// ffmpeg 拉远端视频的硬上限:中 #4 防 disk-fill DoS。
/// 默认 = `MAX_FRAMES_VIDEO` 配的视频字节上限,可通过 cfg.video_limit_bytes 覆盖。
const DEFAULT_VIDEO_FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

/// 用 ffmpeg 把远程 http(s) URL 一次性拉到本地 mp4(支持 HLS/DASH/MP4/WEBM)。
/// 返回本地文件路径 + 落盘字节数。
///
/// 与 `run_job` 的 `spawn_ffmpeg_to_local` 不同:这里要等 ffmpeg 跑完才能
/// 让 core 拿确定性大小的文件去解码,所以同步 `child.wait()`。
/// 中 #4:加 `-fs` 上限 + `run_ffmpeg_wait_with_timeout`,防无尽 HLS 灌爆
/// tmp 磁盘、占死一个并发槽位。
fn fetch_remote_video_to_local(
    url: &str,
    work_dir: &std::path::Path,
    max_bytes: u64,
) -> std::io::Result<(std::path::PathBuf, u64)> {
    use std::process::{Command, Stdio};
    let out_path = work_dir.join("input.mp4");
    let status = run_ffmpeg_wait_with_timeout(
        Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_delay_max",
                "5",
                "-timeout",
                "30000000", // 30s 单连接超时(微秒)
                "-i",
                url,
                "-c",
                "copy", // 优先 copy(快),失败时 ffmpeg 自动回退转码
                "-f",
                "mp4",
                "-movflags",
                "+faststart",
            ])
            .args(["-fs", &max_bytes.to_string()]) // 超 max_bytes 立即停止写入
            .arg(&out_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null()),
        DEFAULT_VIDEO_FETCH_TIMEOUT,
    )
    .map_err(|e| std::io::Error::other(format!("ffmpeg spawn: {e}")))?;
    if !status.success() {
        return Err(std::io::Error::other(
            "ffmpeg video fetch failed (remote URL not retrievable, unsupported format, or exceeded size limit)",
        ));
    }
    if !out_path.is_file() {
        return Err(std::io::Error::other("ffmpeg produced no output file"));
    }
    let size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    if size == 0 {
        return Err(std::io::Error::other("ffmpeg produced 0-byte file"));
    }
    Ok((out_path, size))
}

/// `std::process::Command` 包装版:与 `run_ffmpeg_with_timeout_blocking` 同
/// 思路,但走 `Stdio` 已设的 Command,只负责 wait 循环 + 超时 kill。
/// 抽出它是因为 import_video_url 走自己配的 stdin/stdout/stderr。
fn run_ffmpeg_wait_with_timeout(
    cmd: &mut std::process::Command,
    timeout: std::time::Duration,
) -> std::io::Result<std::process::ExitStatus> {
    let mut child = cmd.spawn()?;
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("ffmpeg wait timeout after {timeout:?}"),
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => {
                let _ = child.kill();
                return Err(e);
            }
        }
    }
}

/// `POST /api/import/video` — multipart upload,直接传视频文件。
/// 与 `/api/jobs/video` 行为等价,但响应额外带 S3 LAN URL。
async fn import_video(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    mp: Multipart,
) -> Response {
    let resp = handle_upload(state.clone(), mp, JobKind::Video).await;
    if !resp.status().is_success() {
        return resp;
    }
    // 复用 handle_upload 的 JSON `{job_id}` 响应:把 id 拿出来重新包一份
    // 带 LAN URL 的导入响应。
    let body_bytes = resp.into_body();
    let body_bytes = axum::body::to_bytes(body_bytes, 4096)
        .await
        .unwrap_or_default();
    let parsed: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
    let id = parsed
        .get("job_id")
        .and_then(|v| v.as_str())
        .map(String::from);
    let Some(id) = id else {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, "missing job_id");
    };
    let job = match state.get(&id) {
        Some(j) => j,
        None => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "job vanished"),
    };
    // 原始文件还没落 S3 时(run_job 还没跑到那一步),URL 为 null;
    // 前端可在轮询 `/api/import/{id}/urls` 拿最新值。
    let key = job
        .original_media_key
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    import_response(&id, JobKind::Video, key.as_deref(), None, job.status())
}

/// `POST /api/import/video-url` — JSON body `{url, algo?}`。
/// 服务端用 ffmpeg 把远程视频拉下来,当作一次性视频任务处理。
async fn import_video_url(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Json(req): Json<ImportVideoUrlReq>,
) -> Response {
    let url = req.url.trim().to_string();
    if url.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "url is required");
    }
    // 只允许 http(s);rtsp/file 走 /api/jobs/stream(语义是"持续监听")。
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "url must be http(s):// for video import; use /api/jobs/stream for rtsp/file",
        );
    }
    // 简单的 SSRF 防御:不允许内网 loopback / link-local / private 地址。
    // 平台本身是 LAN 部署,所以这个限制相对宽松 — 仍挡掉 169.254 / IPv6 link-local / 0.0.0.0。
    // 安全 #1+#2+#3:`::ffff:127.0.0.1` / `%31%32%37.0.0.1` / `user:pass@127.0.0.1`
    // 先规范化再 `is_blocked_host`,挡得住 SSRF 绕过。
    if let Some(host) = url_authority_host(&url) {
        match normalize_host_for_check(host) {
            Some(norm) if is_blocked_host(norm.as_str()) => {
                return error_response(StatusCode::FORBIDDEN, "url host is in a blocked range");
            }
            None => {
                return error_response(
                    StatusCode::FORBIDDEN,
                    "url host contains forbidden characters",
                );
            }
            _ => {}
        }
    }

    let job = match state.create(JobKind::Video, url.clone()) {
        Ok(j) => j,
        Err(e) => return error_response(StatusCode::TOO_MANY_REQUESTS, &e.to_string()),
    };
    let id = job.id.clone();
    state.set_original_input(&id, url.clone());
    if let Some(a) = req.algo {
        state.set_algo_override(&id, a);
    }

    // 用 ffmpeg 拉视频:放到工作目录里(不立刻落 S3 — 让 run_job 落,失败时
    // 工作目录会被 finalize 清理)。这一步同步等,失败立即回 400,避免建空 job。
    let work_dir = state.cfg.tmp_dir.join(&id);
    if let Err(e) = std::fs::create_dir_all(&work_dir) {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("tmp dir: {e}"));
    }
    let (path, size) = match tokio::task::spawn_blocking({
        let url = url.clone();
        let work = work_dir.clone();
        // 中 #4:把视频上限字节传给 ffmpeg `-fs`,防止无尽 HLS / 直播源
        // 把 tmp 磁盘灌满 + 占死一个并发槽位。沿用上传大小限制做默认值。
        let max_bytes = state.cfg.video_limit_bytes;
        move || fetch_remote_video_to_local(&url, &work, max_bytes)
    })
    .await
    {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            // 创建了 job 但拉取失败,不进入 run_job:回退 queued 计数,并清理
            // 可能残留了部分 input.mp4 的工作目录(此路径 run_job/finalize 不会执行)。
            discard_created_job(&state, &id);
            let _ = std::fs::remove_dir_all(&work_dir);
            return error_response(StatusCode::BAD_GATEWAY, &format!("fetch remote video: {e}"));
        }
        Err(e) => {
            discard_created_job(&state, &id);
            let _ = std::fs::remove_dir_all(&work_dir);
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("fetch join: {e}"),
            );
        }
    };

    // 启动检测任务。
    state.spawn_run(job.clone(), path.to_string_lossy().to_string());

    // 第一次响应:先告诉前端"已开始处理";原始文件尚未落 S3,等前端轮询
    // `/api/import/{id}/urls` 拿最终的 LAN URL。
    Json(serde_json::json!({
        "ok": true,
        "job_id": id,
        "kind": JobKind::Video,
        "status": JobStatus::Queued,
        "fetched_bytes": size,
        "original_url": null,
        "cover_url": null,
        "stream_url": format!("/api/jobs/{}/events", id),
        "urls_endpoint": format!("/api/import/{}/urls", id),
    }))
    .into_response()
}

#[derive(Deserialize, Default)]
struct ImportVideoUrlReq {
    url: String,
    /// 可选 per-job 算法覆盖(haar/cnn/...);留空走 env/默认。
    algo: Option<String>,
}

/// `GET /api/import/{id}/urls` — 轻量级"拿 S3 LAN URL"端点。
/// 前端在拿到 job_id 后用 setInterval 轮询;命中本地内存缓存(无锁),
/// 不走任何 DB。
async fn import_urls(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Path(id): Path<String>,
) -> Response {
    let Some(job) = state.get(&id) else {
        return error_response(StatusCode::NOT_FOUND, "no such job");
    };
    let original = job
        .original_media_key
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    // 找第一张标注帧作 cover_key(供 /api/jobs 已有的 SSE 流复用)。
    let cover = job
        .frames
        .lock()
        .unwrap()
        .iter()
        .find_map(|f| f.annotated_key.clone())
        .or_else(|| original.clone());
    Json(serde_json::json!({
        "ok": true,
        "job_id": id,
        "kind": job.kind,
        "status": job.status(),
        "original_url": media_lan_url(original.as_deref()),
        "cover_url": media_lan_url(cover.as_deref()),
        "media_path": media_lan_path(original.as_deref()),
        "stream_url": format!("/api/jobs/{}/events", id),
        "frame_count": job.frames.lock().unwrap_or_else(|e| e.into_inner()).len(),
        "is_terminal": matches!(
            job.status(),
            JobStatus::Done | JobStatus::Cancelled | JobStatus::Error
        ),
    }))
    .into_response()
}

/// 简易 SSRF 拦截:挡掉 loopback / link-local / metadata。
/// 真正的内网 IP 不挡 — 平台本身就在内网,放行 192.168/10/172 段。
///
/// `host` 接受三种写法:
/// 1) `[ipv6]:port` — 剥 `[]`,得到 ipv6 字面量
/// 2) `host:port`   — 单冒号,取 `:` 之前
///
/// 把 URL 里抽出的 host 规范化为 `is_blocked_host` 能直接判断的形式:
/// 1. percent-decode (`%31%32%37.0.0.1` → `127.0.0.1`);
/// 2. IPv4-mapped IPv6 (`[::ffff:127.0.0.1]` → `127.0.0.1`,再走 IPv4 规则);
/// 3. 拒绝含非 ASCII 字符的 host(ffmpeg 会 IDN 解码,字符串规则看不到);
/// 4. 含 `..`(截断路径 / DNS rebinding 候选)直接拒。
///
/// 安全 #1+#2:闭包见审查报告;`is_blocked_host` 收到的是规范化后的 host,
/// `::ffff:127.0.0.1` / `%31%32%37.0.0.1` 不再绕开。
fn normalize_host_for_check(host: &str) -> Option<String> {
    if host.is_empty() || host.contains("..") {
        return None;
    }
    if !host.is_ascii() {
        return None;
    }
    let mut out = String::with_capacity(host.len());
    let bytes = host.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (
                (bytes[i + 1] as char).to_digit(16),
                (bytes[i + 2] as char).to_digit(16),
            ) {
                out.push(((hi << 4) | lo) as u8 as char);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    if let Some(stripped) = out.strip_prefix('[') {
        let inner = stripped.split(']').next().unwrap_or(stripped);
        if let Some(mapped) = inner.strip_prefix("::ffff:") {
            return Some(mapped.to_string());
        }
        return Some(format!("[{inner}]"));
    }
    // 兜底:有些客户端在 `url_authority_host` 之后又经过其他路径削掉了括号,
    // 直接拿到裸 `::ffff:127.0.0.1`。依旧走 IPv4 规则。
    if let Some(mapped) = out.strip_prefix("::ffff:") {
        return Some(mapped.to_string());
    }
    Some(out)
}
/// 3) 裸 ipv6 字面量 (`::1` / `fe80::1`) — 多于一个冒号即视为 ipv6
fn is_blocked_host(host: &str) -> bool {
    let h = if let Some(stripped) = host.strip_prefix('[') {
        // `[ipv6]:port` → 第一个 `]` 之前
        let inner = stripped.split(']').next().unwrap_or(stripped);
        // 安全 #1(防御纵深):`[::ffff:127.0.0.1]` 直接给到本函数也要走 IPv4 规则。
        if let Some(mapped) = inner.strip_prefix("::ffff:") {
            return is_blocked_host(mapped);
        }
        inner
    } else if let Some(stripped) = host.strip_prefix("::ffff:") {
        // 裸 `::ffff:127.0.0.1` 同样剥映射段重判。
        return is_blocked_host(stripped);
    } else if host.matches(':').count() > 1 {
        // 多冒号 → 裸 ipv6
        host
    } else {
        // 单冒号 → host:port
        host.split(':').next().unwrap_or(host)
    };
    let h = h.to_ascii_lowercase();
    if matches!(h.as_str(), "localhost" | "0.0.0.0" | "::" | "::1") {
        return true;
    }
    // IPv6 loopback ::1 and unspecified :: matched above. Block link-local
    // fe80::/10 by the first hextet's top 10 bits, instead of the old
    // `strip_prefix("fe") + len==1` hack which missed the full form
    // "fe80::1" (it only matched the compressed "fe8" spelling).
    if h.contains(':') {
        if let Some(hextet) = h.split(':').next() {
            if let Ok(v) = u16::from_str_radix(hextet, 16) {
                // fe80::/10: top 10 bits == 1111111010 (0b11_1111_1010)
                if (v >> 6) == 0b11_1111_1010 {
                    return true;
                }
            }
        }
    }
    // IPv4 loopback 127.0.0.0/8
    if h.starts_with("127.") {
        return true;
    }
    // IPv4 link-local 169.254.0.0/16 + cloud metadata 169.254.169.254
    if h.starts_with("169.254.") {
        return true;
    }
    // 中 #7:数字式 IPv4 表示(2130706433, 0x7f000001, 0177.0.0.1 等),
    // 上面字符串匹配看不到,要在解析层挡。统一展开成四段数字后照同样规则。
    if let Some(octets) = parse_numeric_ipv4(&h) {
        // 中 #7:数字式 IPv4 (2130706433, 0x7f000001, 0177.0.0.1 等) 表达
        // 只挡"绕回本机 / link-local / metadata":10/172/192.168 走的是产品
        // 显式用例(LAN 摄像头),放行;127/169.254/0.0.0.0/::1 仍然挡。
        return match octets[0] {
            0 => true,               // 0.0.0.0/8
            127 => true,             // 127.0.0.0/8 loopback
            169 => octets[1] == 254, // 169.254.0.0/16 link-local + metadata
            _ => false,
        };
    }
    false
}

/// 把数字形式 IPv4 (`2130706433`, `0x7f000001`, `0177.0.0.1`) 解析成
/// `[u8; 4]`;非数字形式返回 None;只接受"全数字"或"x.. hex"或"0.. octal"
/// 表示法,常规 `127.0.0.1` 由 starts_with 链处理。
fn parse_numeric_ipv4(host: &str) -> Option<[u8; 4]> {
    if !host
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'x' || b == b'X')
    {
        return None;
    }
    let mut octets = [0u8; 4];
    let mut i = 0;
    if host.contains('.') {
        // 点分(可能首段是 hex / octal / dec):逐段解析。
        for seg in host.split('.') {
            if i >= 4 || seg.is_empty() {
                return None;
            }
            let v = parse_int_segment(seg)?;
            if v > 0xff {
                return None;
            }
            octets[i] = v as u8;
            i += 1;
        }
        if i != 4 {
            return None;
        }
        Some(octets)
    } else {
        // 单数字:32 位整数拆字节。
        let v = parse_int_segment(host)?;
        octets[0] = ((v >> 24) & 0xff) as u8;
        octets[1] = ((v >> 16) & 0xff) as u8;
        octets[2] = ((v >> 8) & 0xff) as u8;
        octets[3] = (v & 0xff) as u8;
        Some(octets)
    }
}

/// `0x7f` → 0x7f (hex);`0177` → 0x7f (octal,前导 0 视为 octal);
/// 其它 → u8 dec。
fn parse_int_segment(seg: &str) -> Option<u32> {
    if seg.len() > 1 && (seg.starts_with("0x") || seg.starts_with("0X")) {
        u32::from_str_radix(&seg[2..], 16).ok()
    } else if seg.len() > 1 && seg.starts_with('0') && seg.bytes().all(|b| b.is_ascii_digit()) {
        u32::from_str_radix(seg, 8).ok()
    } else {
        seg.parse::<u32>().ok()
    }
}

// ---------------------------------------------------------------------------
// 缓存辅助
// ---------------------------------------------------------------------------

/// JSON 缓存响应(带 ETag)。前端用 If-None-Match 命中时直接 304。
fn cached_json_response_with_etag(bytes: Vec<u8>, etag: &str) -> Response {
    let mut resp = ([(header::CONTENT_TYPE, "application/json")], bytes).into_response();
    if let Ok(hv) = HeaderValue::from_str(etag) {
        resp.headers_mut().insert(header::ETAG, hv);
    }
    resp
}

/// 简易 FNV-1a 64-bit hash,避免引入 `xxhash-rust` 等额外依赖。
/// 用于 ETag 派生 — 抗碰撞足够,不必加密学安全。
fn fxhash_short(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 比较客户端 If-None-Match 头(逗号分隔多值,strip W/)和服务端 ETag。
fn if_none_match_matches(client_hdr: Option<&str>, server_etag: &str) -> bool {
    let Some(hdr) = client_hdr else {
        return false;
    };
    fn strip(s: &str) -> &str {
        s.strip_prefix("W/").unwrap_or(s).trim()
    }
    let server_stripped = strip(server_etag);
    hdr.split(',').any(|v| strip(v.trim()) == server_stripped)
}

// ============================================================================
// 埋点摘要 / 近期事件 — 给前端 dashboard 用的轻量聚合
// (复用 persist::Db;无 DB 时返回空结构,前端降级显示)
// ============================================================================

#[derive(Deserialize)]
struct TelemetryQuery {
    /// 摘要窗口(秒);默认 24h(86400s)。仅供 summary 用。
    window_secs: Option<i64>,
    /// recent 限制条数;默认 50,上限 500。
    limit: Option<usize>,
}

async fn telemetry_summary(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Query(q): Query<TelemetryQuery>,
) -> Response {
    let window_secs = q.window_secs.unwrap_or(86_400).clamp(60, 7 * 86_400);
    let Some(db_pool) = state.db.pool.clone() else {
        return Json(serde_json::json!({
            "ok": true, "enabled": false, "window_secs": window_secs,
            "buckets": [], "total": 0,
        }))
        .into_response();
    };
    // 跨数据库连接跑聚合查询;若失败返回 200 + enabled:false,前端降级。
    let result = sqlx::query(
        "SELECT name, COUNT(*) AS n
         FROM telemetry
         WHERE received_ms > (EXTRACT(epoch FROM now())::bigint * 1000 - $1)
         GROUP BY name
         ORDER BY n DESC
         LIMIT 50",
    )
    .bind(window_secs * 1000)
    .fetch_all(&db_pool)
    .await;
    match result {
        Ok(rows) => {
            use sqlx::Row;
            let buckets: Vec<serde_json::Value> = rows
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "name": r.get::<String, _>("name"),
                        "count": r.get::<i64, _>("n"),
                    })
                })
                .collect();
            let total: i64 = rows.iter().map(|r| r.get::<i64, _>("n")).sum();
            Json(serde_json::json!({
                "ok": true,
                "enabled": true,
                "window_secs": window_secs,
                "buckets": buckets,
                "total": total,
            }))
            .into_response()
        }
        Err(e) => {
            eprintln!("[telemetry] summary query failed: {e}");
            Json(serde_json::json!({
                "ok": true, "enabled": false, "error": e.to_string(),
                "window_secs": window_secs, "buckets": [], "total": 0,
            }))
            .into_response()
        }
    }
}

async fn telemetry_recent(
    State((state, _)): State<(Arc<JobRegistry>, ResponseCaches)>,
    Query(q): Query<TelemetryQuery>,
) -> Response {
    let limit = q.limit.unwrap_or(50).clamp(1, 500);
    let Some(db_pool) = state.db.pool.clone() else {
        return Json(serde_json::json!({
            "ok": true, "enabled": false, "events": [],
        }))
        .into_response();
    };
    let result = sqlx::query(
        "SELECT name, ts_ms, session, path, props
         FROM telemetry
         ORDER BY id DESC
         LIMIT $1",
    )
    .bind(limit as i64)
    .fetch_all(&db_pool)
    .await;
    match result {
        Ok(rows) => {
            use sqlx::Row;
            let events: Vec<serde_json::Value> = rows
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "name": r.get::<String, _>("name"),
                        "ts_ms": r.get::<i64, _>("ts_ms"),
                        "session": r.get::<Option<String>, _>("session"),
                        "path": r.get::<Option<String>, _>("path"),
                        "props": r.try_get::<Option<serde_json::Value>, _>("props").unwrap_or(None),
                    })
                })
                .collect();
            Json(serde_json::json!({
                "ok": true,
                "enabled": true,
                "events": events,
            }))
            .into_response()
        }
        Err(e) => {
            eprintln!("[telemetry] recent query failed: {e}");
            Json(serde_json::json!({
                "ok": true, "enabled": false, "error": e.to_string(),
                "events": [],
            }))
            .into_response()
        }
    }
}

#[cfg(test)]
mod import_tests {
    use super::*;

    #[test]
    fn url_encode_path_keeps_slashes_and_dots() {
        assert_eq!(
            url_encode_path("jobs/abc/original.mp4"),
            "jobs/abc/original.mp4"
        );
        assert_eq!(url_encode_path("a b?c"), "a%20b%3Fc");
        // 已经编码的百分号要二次 encode(避免把 %3F 拆开)
        assert_eq!(url_encode_path("a%20b"), "a%2520b");
    }

    #[test]
    fn sse_terminal_detection_is_typed() {
        assert!(sse_payload_is_terminal(r#"{"type":"done"}"#));
        assert!(sse_payload_is_terminal(r#"{"type":"error","message":"x"}"#));
        assert!(sse_payload_is_terminal(
            r#"{"status":"x","type":"cancelled"}"#
        ));
        // 非终态 / 非 JSON / 缺字段 → false
        assert!(!sse_payload_is_terminal(r#"{"type":"frame"}"#));
        assert!(!sse_payload_is_terminal("not json"));
        assert!(!sse_payload_is_terminal(r#"{"no":"type"}"#));
        // 带空格的 JSON 也能识别(旧子串匹配会漏)
        assert!(sse_payload_is_terminal(r#"{ "type": "done" }"#));
    }

    #[test]
    fn url_authority_host_parses_schemes() {
        assert_eq!(
            url_authority_host("rtsp://192.168.1.10:554/cam"),
            Some("192.168.1.10")
        );
        // userinfo 必须剥掉
        assert_eq!(
            url_authority_host("https://user:pass@example.com/path"),
            Some("example.com")
        );
        // 无 port 的 plain host
        assert_eq!(
            url_authority_host("http://localhost:8080"),
            Some("localhost")
        );
        // bracketed IPv6
        assert_eq!(url_authority_host("rtsp://[fe80::1]:554/"), Some("fe80::1"));
        // 畸形:空 host / 无 scheme
        assert!(url_authority_host("rtsp:///garbage").is_none());
        assert!(url_authority_host("not-a-url").is_none());
    }

    #[test]
    fn is_blocked_host_blocks_loopback_and_metadata() {
        assert!(is_blocked_host("localhost"));
        assert!(is_blocked_host("127.0.0.1:8080"));
        assert!(is_blocked_host("127.0.0.1"));
        assert!(is_blocked_host("[::1]"));
        assert!(is_blocked_host("::1"));
        assert!(is_blocked_host("[fe80::1]"));
        assert!(is_blocked_host("fe80::1"));
        // fe80::/10 spans fe80..febf — full (non-compressed) spellings too
        assert!(is_blocked_host("[febf:1234::5]:8080"));
        assert!(is_blocked_host("fea0::abcd"));
        // unique-local fc00::/7 and global IPv6 must NOT be link-local
        assert!(!is_blocked_host("fc00::1"));
        assert!(!is_blocked_host("2606:4700:4700::1111"));
        assert!(is_blocked_host("169.254.169.254"));
        assert!(is_blocked_host("169.254.0.5"));
        // 内网 IP 放行(平台本身是 LAN 部署)
        assert!(!is_blocked_host("192.168.1.10"));
        assert!(!is_blocked_host("10.0.0.5"));
        assert!(!is_blocked_host("172.16.0.1"));
        // 公网域名放行
        assert!(!is_blocked_host("example.com"));
    }

    /// 安全 #1+#2:`::ffff:127.0.0.1` IPv4-mapped 绕过;`%31%32%37.0.0.1`
    /// percent-encoded 绕过。`normalize_host_for_check` 把这些先解出来,
    /// `is_blocked_host` 再判。
    #[test]
    fn normalize_host_blocks_v4_mapped_and_percent_encoded_bypass() {
        // IPv4-mapped IPv6 loopback
        let h = normalize_host_for_check("[::ffff:127.0.0.1]").unwrap();
        assert!(
            is_blocked_host(&h),
            "::ffff:127.0.0.1 should be blocked: {h}"
        );
        // IPv4-mapped cloud metadata
        let h = normalize_host_for_check("[::ffff:169.254.169.254]").unwrap();
        assert!(
            is_blocked_host(&h),
            "::ffff:169.254.169.254 should be blocked: {h}"
        );
        // IPv4-mapped 仍允许的内网(LAN 产品用例)
        let h = normalize_host_for_check("[::ffff:192.168.1.10]").unwrap();
        assert!(
            !is_blocked_host(&h),
            "::ffff:192.168.1.10 should NOT be blocked: {h}"
        );
        // percent-encoded 127.0.0.1(`%31%32%37.0.0.1` 解码 = `127.0.0.1`)
        let h = normalize_host_for_check("%31%32%37.0.0.1").unwrap();
        assert!(
            is_blocked_host(&h),
            "%31%32%37.0.0.1 should be blocked: {h}"
        );
        // 非 ASCII(IDN)拒收
        assert!(normalize_host_for_check("①②⑦.0.0.1").is_none());
        // `..` 拒收
        assert!(normalize_host_for_check("127..0.0.1").is_none());
    }

    /// 安全 #1 防御纵深:`is_blocked_host` 直接看到 `[::ffff:127.0.0.1]` /
    /// 裸 `::ffff:127.0.0.1` 也应该挡 — 之前依赖 `normalize_host_for_check`
    /// 处理,但 normalize 漏覆盖就会回归。手动 call `is_blocked_host` 验证。
    #[test]
    fn is_blocked_host_handles_ipv4_mapped_directly() {
        assert!(is_blocked_host("[::ffff:127.0.0.1]"));
        assert!(is_blocked_host("::ffff:127.0.0.1"));
        assert!(is_blocked_host("[::ffff:169.254.169.254]"));
        assert!(is_blocked_host("::ffff:169.254.169.254"));
        assert!(!is_blocked_host("[::ffff:192.168.1.10]"));
    }

    #[test]
    fn media_lan_path_strips_scheme() {
        assert_eq!(
            media_lan_path(Some("s3://jobs/abc/original.mp4")).as_deref(),
            Some("jobs/abc/original.mp4")
        );
        assert_eq!(
            media_lan_path(Some("local://jobs/abc/original.mp4")).as_deref(),
            Some("jobs/abc/original.mp4")
        );
        assert_eq!(
            media_lan_path(Some("inline://jobs/abc/original.mp4")).as_deref(),
            Some("jobs/abc/original.mp4")
        );
        assert_eq!(media_lan_path(None), None);
    }

    #[test]
    fn if_none_match_handles_weak_etags_and_multi() {
        let s = "W/\"abc-123\"";
        assert!(if_none_match_matches(Some("W/\"abc-123\""), s));
        assert!(if_none_match_matches(Some("\"abc-123\""), s));
        assert!(if_none_match_matches(Some("W/\"other\", W/\"abc-123\""), s));
        assert!(!if_none_match_matches(Some("W/\"different\""), s));
        assert!(!if_none_match_matches(None, s));
    }

    #[test]
    fn fxhash_short_distinguishes_inputs() {
        let h1 = fxhash_short(b"hello");
        let h2 = fxhash_short(b"world");
        let h3 = fxhash_short(b"hello");
        assert_ne!(h1, h2);
        assert_eq!(h1, h3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_header_basic() {
        // 普通 start-end
        assert_eq!(
            parse_range_header("bytes=0-499", 1000),
            Some((0, Some(499)))
        );
        assert_eq!(
            parse_range_header("bytes=500-999", 1000),
            Some((500, Some(999)))
        );
    }

    #[test]
    fn range_header_open_ended() {
        // start- (到 EOF)
        assert_eq!(parse_range_header("bytes=100-", 1000), Some((100, None)));
    }

    #[test]
    fn range_header_suffix() {
        // -N:最后 N 字节
        assert_eq!(
            parse_range_header("bytes=-200", 1000),
            Some((800, Some(999)))
        );
        // N 超过总长:从头开始
        assert_eq!(
            parse_range_header("bytes=-2000", 1000),
            Some((0, Some(999)))
        );
    }

    #[test]
    fn range_header_clamps_end_to_eof() {
        // end 超过 total-1 → 钳到 total-1
        assert_eq!(
            parse_range_header("bytes=900-2000", 1000),
            Some((900, Some(999)))
        );
    }

    #[test]
    fn range_header_unsatisfiable() {
        // start >= total → u64::MAX 标记(调用方回 416)
        assert_eq!(
            parse_range_header("bytes=1000-", 1000),
            Some((u64::MAX, None))
        );
        assert_eq!(
            parse_range_header("bytes=5000-6000", 1000),
            Some((u64::MAX, None))
        );
    }

    #[test]
    fn range_header_multi_range_and_malformed_fall_back_to_full() {
        // 多区间 → None(200 全量)
        assert_eq!(parse_range_header("bytes=0-1,5-6", 1000), None);
        // 非 bytes 单位 / 非法数字 → None
        assert_eq!(parse_range_header("items=0-1", 1000), None);
        assert_eq!(parse_range_header("bytes=abc-", 1000), None);
        assert_eq!(parse_range_header("", 1000), None);
    }

    #[tokio::test]
    async fn local_range_stream_returns_exact_slice() {
        use tokio::io::AsyncReadExt;
        let dir = std::env::temp_dir().join("rsface-api-test-range");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.bin");
        let data: Vec<u8> = (0u8..=255).collect(); // 256 bytes
        std::fs::write(&path, &data).unwrap();
        let (mut reader, total) = read_local_range_stream(&path, 10, Some(20)).await.unwrap();
        assert_eq!(total, 256);
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await.unwrap();
        assert_eq!(buf, &data[10..=20]);
        // 开放区间到 EOF
        let (mut reader2, _) = read_local_range_stream(&path, 250, None).await.unwrap();
        let mut buf2 = Vec::new();
        reader2.read_to_end(&mut buf2).await.unwrap();
        assert_eq!(buf2, &data[250..]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
