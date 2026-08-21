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
//! - `GET  /api/metrics`          平台实时指标(前端 KPI 栏轮询)
//! - `GET  /media/{key}`          S3/本地 媒体代理(支持 Range/206)

use crate::jobs::{JobKind, JobRegistry, JobStatus};
use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::convert::Infallible;
use std::io::BufReader;
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

pub fn router(state: Arc<JobRegistry>) -> Router {
    // 上传上限分层:图片(默认 50MB)/ 视频(默认 2GB)。
    // 全局不再用 1GB 统一限制;其它路由(JSON/SSE)走 axum 默认 2MB。
    let img_limit = state.cfg.upload_limit_image;
    let video_limit = state.cfg.upload_limit_video;
    let cors_origin = state.cfg.cors_allow_origin.clone();
    Router::new()
        .route("/", get(index))
        .route("/api/health", get(health))
        .route("/api/config", get(config_info))
        .route("/api/metrics", get(metrics))
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
        .route("/api/jobs/{id}/retry", post(retry_job))
        .route("/media/{*key}", get(media))
        .route("/{file}", get(static_file))
        .layer(axum::middleware::from_fn(move |req, next| {
            cors_middleware(req, next, cors_origin.clone())
        }))
        .with_state(state)
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
    headers.insert(
        axum::http::HeaderName::from_static("access-control-max-age"),
        hv("86400"),
    );
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok", "service": "rsface-platform"}))
}

/// 报告当前检测器模式 + 权重/级联文件状态 + 可用算法列表。
///
/// 模式选择规则(在 `jobs::build_detector` 中):
/// - `RSFACE_ALGO` 环境变量显式选择(`haar` / `cnn` / `yunet` / `mtcnn` / `hog`)
/// - 否则:`use_cnn=true` 或 `cnn_weights` 路径已设置 → `"cnn"`
/// - 否则 → `"haar"`
///
/// 权重状态(仅 cnn 模式有意义):
/// - `template` — 未指定权重文件,使用 core 内置 hand-crafted 模板
/// - `available` — 路径已设置且文件存在
/// - `missing`  — 路径已设置但文件不存在(run_job 启动时会失败)
async fn config_info(State(state): State<Arc<JobRegistry>>) -> Json<serde_json::Value> {
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
    Json(serde_json::json!({
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
    }))
}

async fn index(State(state): State<Arc<JobRegistry>>) -> Response {
    serve_static(&state.cfg.web_dir, "index.html").await
}

async fn static_file(State(state): State<Arc<JobRegistry>>, Path(file): Path<String>) -> Response {
    serve_static(&state.cfg.web_dir, &file).await
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
    match tokio::fs::read(&path).await {
        Ok(bytes) => {
            let ct = content_type_for(file);
            ([(header::CONTENT_TYPE, ct)], bytes).into_response()
        }
        Err(_) => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
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
    State(state): State<Arc<JobRegistry>>,
    Query(q): Query<ListJobsQuery>,
) -> Json<serde_json::Value> {
    let all = state.list();
    let total = all.len();
    // 无 limit/offset:保持旧版响应 shape(只有 jobs 数组,向后兼容)。
    if q.limit.is_none() && q.offset.is_none() {
        let jobs: Vec<serde_json::Value> = all.iter().map(|j| j.summary()).collect();
        return Json(serde_json::json!({"jobs": jobs}));
    }
    let offset = q.offset.unwrap_or(0).min(total);
    let limit = q.limit.unwrap_or(total.saturating_sub(offset));
    let jobs: Vec<serde_json::Value> = all
        .iter()
        .skip(offset)
        .take(limit)
        .map(|j| j.summary())
        .collect();
    Json(serde_json::json!({
        "jobs": jobs,
        "total": total,
        "limit": limit,
        "offset": offset,
    }))
}

/// `GET /api/jobs/stats`:按算法(haar/cnn/yunet/mtcnn/hog)聚合
/// 成功/失败/取消/平均耗时/检出数。queued 未定算法的归入 `pending`。
async fn job_stats(State(state): State<Arc<JobRegistry>>) -> Json<serde_json::Value> {
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
    Json(serde_json::json!({
        "algos": algos,
        "total_jobs": total_jobs,
    }))
}

/// `GET /api/metrics`:平台实时指标(前端 KPI 栏每 2s 轮询)。
/// 字段与 web/app.js kpi.refresh() 的读取保持兼容:
/// running / max_concurrency / total_frames_processed / total_frames_with_face /
/// total_detections / errored / cancelled / mode。
/// live_fps_max / gpu_pct / cascade_pass_rate 平台侧暂无数据源,返回 0
/// (前端对 0 有降级显示,不会报错)。
async fn metrics(State(state): State<Arc<JobRegistry>>) -> Json<serde_json::Value> {
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
        // 每个 job 只短持一次 stats 锁,不在 await 点持有。
        let st = j.stats.lock().unwrap().clone();
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
    Json(serde_json::json!({
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
    }))
}

async fn job_detail(State(state): State<Arc<JobRegistry>>, Path(id): Path<String>) -> Response {
    match state.get(&id) {
        Some(job) => {
            let mut v = job.summary();
            v["frames"] = serde_json::to_value(&*job.frames.lock().unwrap()).unwrap_or_default();
            Json(v).into_response()
        }
        None => (StatusCode::NOT_FOUND, "no such job").into_response(),
    }
}

async fn cancel_job(State(state): State<Arc<JobRegistry>>, Path(id): Path<String>) -> Response {
    match state.get(&id) {
        Some(job) => {
            job.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            Json(serde_json::json!({"ok": true})).into_response()
        }
        None => (StatusCode::NOT_FOUND, "no such job").into_response(),
    }
}

async fn delete_job(State(state): State<Arc<JobRegistry>>, Path(id): Path<String>) -> Response {
    state.request_cancel(&id);
    let existed = state.remove(&id);
    let db = state.db.clone();
    let id_db = id.clone();
    tokio::spawn(async move {
        db.delete_job(&id_db).await;
    });
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
async fn batch_ops(State(state): State<Arc<JobRegistry>>, Json(req): Json<BatchReq>) -> Response {
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
                    let frames = j.frames.lock().unwrap();
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

async fn retry_job(State(state): State<Arc<JobRegistry>>, Path(id): Path<String>) -> Response {
    let original = {
        let Some(j) = state.get(&id) else {
            return error_response(StatusCode::NOT_FOUND, "no such job");
        };
        let inp = j.original_input.lock().unwrap().clone();
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
    /// Comma-separated algo list, e.g. `haar,cnn,yunet`. Optional —
    /// if missing, runs all 5 available algos.
    #[serde(default)]
    algos: Option<String>,
    /// Frame index for video/stream jobs (defaults to first frame with faces, or 0).
    /// 预留字段:当前实现固定用第一帧;保留解析以兼容已有调用方。
    #[serde(default)]
    #[allow(dead_code)]
    frame: Option<u64>,
}

/// `POST /api/jobs/{id}/compare?algos=haar,cnn,yunet`
/// 对任务的第一张图(或指定 frame)同时跑多个算法,返回每个算法的
/// detection 数、耗时、bounding boxes。前端用它来渲染 5 张并排小图
/// 的"算法对比"视图。
async fn compare_algos(
    State(state): State<Arc<JobRegistry>>,
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

    // 2) 拿到 job 对应的原始媒体字节(S3 优先,失败回退到 local media dir)。
    let job = match state.get(&id) {
        Some(j) => j,
        None => return error_response(StatusCode::NOT_FOUND, "no such job"),
    };
    let media_key = job.original_media_key.lock().unwrap().clone();
    let media_key = match media_key {
        Some(k) => k,
        None => {
            return error_response(
                StatusCode::NOT_FOUND,
                "job has no original media (stream jobs not supported)",
            )
        }
    };
    let bytes: Vec<u8> = {
        if let Some(rest) = media_key.strip_prefix("local://") {
            let path = state.cfg.local_media_dir.join(rest);
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
            let res = tokio::task::spawn_blocking(move || s3.get_object(&owned)).await;
            match res {
                Ok(Ok((b, _ct))) => b,
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

    // 4) 对每个 algo 跑 detect(同步,因为 DetectorKind 不是 Send)。
    //    因为 CnnDetector/Yunet 持有 !Sync scratch,不能在多线程间共享,
    //    所以串行跑,而不是 spawn_blocking.parallel。
    let mut results: Vec<serde_json::Value> = Vec::new();
    for algo in &valid {
        let det = match crate::jobs::build_detector_by_name(algo) {
            Ok(d) => d,
            Err(e) => {
                results.push(serde_json::json!({
                    "algo": algo,
                    "error": e.to_string(),
                }));
                continue;
            }
        };
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

    Json(serde_json::json!({
        "job_id": id,
        "width": gray.width(),
        "height": gray.height(),
        "requested_algos": valid,
        "results": results,
    }))
    .into_response()
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
    // JPG / WebP / other: fall back to ffmpeg → PGM.
    let work_dir = tmp_dir.join(format!("compare-{job_id}"));
    std::fs::create_dir_all(&work_dir)?;
    let in_path = work_dir.join("input.bin");
    let out_path = work_dir.join("out.pgm");
    std::fs::write(&in_path, bytes)?;
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-i"])
        .arg(&in_path)
        .args(["-pix_fmt", "gray", "-f", "image2"])
        .arg(&out_path)
        .output()
        .map_err(|e| std::io::Error::other(format!("ffmpeg spawn: {e}")))?;
    if !status.status.success() {
        return Err(std::io::Error::other(format!(
            "ffmpeg image convert failed: {}",
            String::from_utf8_lossy(&status.stderr)
        )));
    }
    let f = std::fs::File::open(&out_path)?;
    let mut reader = BufReader::new(f);
    rsface::image::codec::read_pgm(&mut reader)
}

async fn upload_image(State(state): State<Arc<JobRegistry>>, mp: Multipart) -> Response {
    handle_upload(state, mp, JobKind::Image).await
}

async fn upload_video(State(state): State<Arc<JobRegistry>>, mp: Multipart) -> Response {
    handle_upload(state, mp, JobKind::Video).await
}

async fn handle_upload(state: Arc<JobRegistry>, mut mp: Multipart, kind: JobKind) -> Response {
    // 分层大小上限(路由层 DefaultBodyLimit 已按 kind 限制,这里再在
    // field 读取时做内存保护:超过即中断,避免把整个 body 缓进内存)。
    let max_bytes = match kind {
        JobKind::Image => state.cfg.upload_limit_image,
        _ => state.cfg.upload_limit_video,
    };
    // 取第一个 file 字段。
    let mut filename = None;
    let mut bytes: Vec<u8> = Vec::new();
    while let Ok(Some(field)) = mp.next_field().await {
        if field.name() != Some("file") {
            continue;
        }
        filename = field.file_name().map(|s| s.to_string());
        match field.bytes().await {
            Ok(b) => {
                if b.len() > max_bytes {
                    return error_response(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        &format!(
                            "upload too large: {} bytes (max {} bytes)",
                            b.len(),
                            max_bytes
                        ),
                    );
                }
                bytes = b.to_vec();
                break;
            }
            Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("read upload: {e}")),
        }
    }
    let Some(name) = filename else {
        return error_response(StatusCode::BAD_REQUEST, "missing 'file' field");
    };
    if bytes.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "empty upload");
    }

    let ext = sanitized_ext(&name, kind);
    let job = match state.create(kind, name.clone()) {
        Ok(j) => j,
        Err(e) => return error_response(StatusCode::TOO_MANY_REQUESTS, &e.to_string()),
    };
    let id = job.id.clone();
    state.set_original_input(&id, name.clone());

    // 落盘(阻塞 IO 放到 blocking 线程)。
    // Image kind:对 PNG / JPG 先 ffmpeg 转 PGM,平台层兜底,core 的 PNG
    // 解码只支持 stored 块不再成为瓶颈;同时把用户的原始字节另行落到
    // local_media_dir/original.{ext},供 web 端的 preview 展示。
    let (path, pre_stored_original) = {
        let cfg = state.cfg.clone();
        let idc = id.clone();
        let kind_l = kind;
        let ext_l = ext.clone();
        let bytes_l = bytes.clone();
        match tokio::task::spawn_blocking(move || -> std::io::Result<(String, Option<String>)> {
            // 1) 用户的原始文件写到 tmp_dir(给 ffmpeg / core 读)
            let raw_path = crate::jobs::save_upload(&cfg, &idc, &ext_l, &bytes_l)?;
            // 2) Image:把 PNG/JPG 转成 PPM(RGB 三通道),core 直接吃
            //    旧实现是 PGM(灰度),导致标注/裁剪全是灰色。
            //    注意:这里用 ppm 还是走 core 的 read_ppm 路径,会同时拿到 gray + rgb。
            if kind_l == JobKind::Image
                && matches!(ext_l.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "webp")
            {
                let work = raw_path.parent().unwrap().to_path_buf();
                let ppm = work.join("input.ppm");
                let status = std::process::Command::new("ffmpeg")
                    .args(["-y", "-loglevel", "error", "-i"])
                    .arg(&raw_path)
                    .args(["-pix_fmt", "rgb24", "-f", "image2"])
                    .arg(&ppm)
                    .output()
                    .map_err(|e| std::io::Error::other(format!("ffmpeg spawn: {e}")))?;
                if !status.status.success() || !ppm.is_file() {
                    return Err(std::io::Error::other(format!(
                        "ffmpeg image→ppm failed: {}",
                        String::from_utf8_lossy(&status.stderr)
                    )));
                }
                // 3) 用户的原始字节落到 local_media_dir,作 preview 用
                let display_key = format!("jobs/{idc}/original.{ext_l}");
                let ct = match ext_l.as_str() {
                    "png" => "image/png",
                    "jpg" | "jpeg" => "image/jpeg",
                    "bmp" => "image/bmp",
                    "webp" => "image/webp",
                    _ => "application/octet-stream",
                };
                let stored =
                    crate::jobs::put_bytes_with_fallback_blocking(&cfg, &display_key, ct, &bytes_l);
                Ok((ppm.to_string_lossy().to_string(), Some(stored)))
            } else {
                Ok((raw_path.to_string_lossy().to_string(), None))
            }
        })
        .await
        {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("upload prep: {e}"),
                )
            }
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("upload join: {e}"),
                )
            }
        }
    };

    // 如果已经在 handle_upload 阶段存了 original,直接更新 job 的 original_media_key,
    // run_job 看到已设置就会跳过重复存储。
    if let Some(stored) = pre_stored_original {
        *job.original_media_key.lock().unwrap() = Some(stored.clone());
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
}

async fn start_stream(
    State(state): State<Arc<JobRegistry>>,
    Json(req): Json<StreamReq>,
) -> Response {
    let url = req.url.trim().to_string();
    if !(url.starts_with("rtsp://")
        || url.starts_with("http://")
        || url.starts_with("https://")
        || url.starts_with("file://")
        || url.starts_with("test://"))
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            "url must be rtsp:// http(s):// file:// or test://",
        );
    }
    let display = url.clone();
    let job = match state.create(JobKind::Stream, display) {
        Ok(j) => j,
        Err(e) => return error_response(StatusCode::TOO_MANY_REQUESTS, &e.to_string()),
    };
    let id = job.id.clone();
    state.set_original_input(&id, url.clone());
    state.spawn_run(job, url);
    Json(serde_json::json!({"job_id": id})).into_response()
}

/// SSE 事件查询参数。`last_event_id` 是 SSE 协议约定的断点续传字段;
/// `?last_event_id=42` 表示客户端已经收到 id=42 之前的所有事件,只重发 >42 的。
#[derive(Default, Deserialize)]
struct EventsQuery {
    last_event_id: Option<u64>,
}

async fn job_events(
    State(state): State<Arc<JobRegistry>>,
    Path(id): Path<String>,
    Query(q): Query<EventsQuery>,
) -> Response {
    let Some(job) = state.get(&id) else {
        return (StatusCode::NOT_FOUND, "no such job").into_response();
    };
    let rx = job.event_tx.subscribe();

    // 起始 event id = 客户端上次收到的;没传则从 0 开始。
    let start_after = q.last_event_id.unwrap_or(0);
    // 把 job 现有的帧当作历史回放,跳过 start_after 之前的。
    let mut seq: u64 = 0;
    let replay: Vec<(u64, String)> = {
        let frames = job.frames.lock().unwrap();
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
                    let done = payload.contains("\"type\":\"done\"")
                        || payload.contains("\"type\":\"error\"")
                        || payload.contains("\"type\":\"cancelled\"");
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
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx_async).map(Ok::<Event, Infallible>);
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

/// 构造 206 Partial Content 响应(动态 Content-Range/Content-Length 需要运行期
/// 字符串,用 builder 逐个 insert 而非静态 header 元组)。
fn range_response(
    code: StatusCode,
    ct: &str,
    cache: &str,
    content_range: &str,
    body: Vec<u8>,
) -> Response {
    let mut resp = (code, body).into_response();
    let headers = resp.headers_mut();
    let _ = headers.try_insert(
        header::CONTENT_TYPE,
        axum::http::HeaderValue::from_str(ct).unwrap_or(axum::http::HeaderValue::from_static(
            "application/octet-stream",
        )),
    );
    let _ = headers.try_insert(
        header::CACHE_CONTROL,
        axum::http::HeaderValue::from_str(cache)
            .unwrap_or(axum::http::HeaderValue::from_static("public, max-age=3600")),
    );
    let _ = headers.try_insert(
        header::ACCEPT_RANGES,
        axum::http::HeaderValue::from_static("bytes"),
    );
    let _ = headers.try_insert(
        header::CONTENT_RANGE,
        axum::http::HeaderValue::from_str(content_range)
            .unwrap_or(axum::http::HeaderValue::from_static("bytes */0")),
    );
    resp
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

/// 用 tokio::fs 按 seek+read 切片(本地媒体 Range 路径,避免整个文件读进内存)。
async fn read_local_range(
    path: &std::path::Path,
    start: u64,
    end_inclusive: Option<u64>,
) -> std::io::Result<(Vec<u8>, u64)> {
    use tokio::io::{AsyncReadExt, AsyncSeekExt};
    let mut f = tokio::fs::File::open(path).await?;
    let total = f.metadata().await?.len();
    if start >= total {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "range start beyond EOF",
        ));
    }
    let end = end_inclusive.unwrap_or(total - 1).min(total - 1);
    let len = end - start + 1;
    f.seek(std::io::SeekFrom::Start(start)).await?;
    let mut buf = vec![0u8; len as usize];
    f.read_exact(&mut buf).await?;
    Ok((buf, total))
}

async fn media(
    State(state): State<Arc<JobRegistry>>,
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
        return (StatusCode::BAD_REQUEST, "bad key").into_response();
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
                    match read_local_range(&local_path, start, end).await {
                        Ok((buf, total)) => {
                            let first = start;
                            let last = start + buf.len() as u64 - 1;
                            return range_response(
                                StatusCode::PARTIAL_CONTENT,
                                ct,
                                cache,
                                &format!("bytes {first}-{last}/{total}"),
                                buf,
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
            Err(_) => (StatusCode::NOT_FOUND, "local object not found").into_response(),
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
                        s3r.get_object_range(&key_r, start, end)
                    })
                    .await;
                    if let Ok(Ok((buf, _total))) = res {
                        let first = start;
                        let last = start + buf.len() as u64 - 1;
                        return range_response(
                            StatusCode::PARTIAL_CONTENT,
                            ct,
                            cache,
                            &format!("bytes {first}-{last}/{total}"),
                            buf,
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
            (StatusCode::NOT_FOUND, "object not found").into_response()
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

fn error_response(code: StatusCode, msg: &str) -> Response {
    (code, Json(serde_json::json!({"error": msg}))).into_response()
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
    async fn local_range_read_returns_exact_slice() {
        let dir = std::env::temp_dir().join("rsface-api-test-range");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.bin");
        let data: Vec<u8> = (0u8..=255).collect(); // 256 bytes
        std::fs::write(&path, &data).unwrap();
        let (buf, total) = read_local_range(&path, 10, Some(20)).await.unwrap();
        assert_eq!(total, 256);
        assert_eq!(buf, &data[10..=20]);
        // 开放区间到 EOF
        let (buf2, _) = read_local_range(&path, 250, None).await.unwrap();
        assert_eq!(buf2, &data[250..]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
