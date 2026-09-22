//! HTTP handlers for the persistent person / face gallery.
//!
//! All routes are thin: they parse the request, delegate to
//! `gallery::GalleryState` for the actual work, and translate errors
//! into the platform's standard `{"error", "error_code", ...}` JSON
//! envelope (see `api::error_response_with`).
//!
//! Routes wired in `api::router`:
//!
//!   GET    /api/persons[?include_archived]
//!   POST   /api/persons                       (create)
//!   GET    /api/persons/{id}
//!   DELETE /api/persons/{id}                  (archive, not destroy)
//!   POST   /api/persons/{id}/faces            (multipart image)
//!   GET    /api/persons/{id}/faces-list
//!   DELETE /api/faces/{id}
//!   POST   /api/identify                      (multipart image → top-k matches)
//!   POST   /api/verify                        (multipart image + label → cosine)

use crate::api::error_response_with;
use crate::api::ResponseCaches;
use crate::gallery::{FaceEnroll, FaceMeta, Person, PersonCreate};
use crate::jobs::{DetectorKind, JobRegistry};
use axum::extract::{Multipart, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rsface::image::codec::read_pgm;
use rsface::image::png::decode_to_gray;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

type AppState = (Arc<JobRegistry>, ResponseCaches);

pub async fn list_persons(
    State((state, _)): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<ListParams>,
) -> Response {
    let persons = state.gallery.list_persons(params.include_archived).await;
    Json(PersonList {
        count: persons.len(),
        persons,
    })
    .into_response()
}

#[derive(Debug, Default, Deserialize)]
pub struct ListParams {
    #[serde(default)]
    pub include_archived: bool,
}

#[derive(Debug, Serialize)]
pub struct PersonList {
    pub count: usize,
    pub persons: Vec<Person>,
}

pub async fn get_person(State((state, _)): State<AppState>, Path(id): Path<String>) -> Response {
    match state.gallery.get_person(&id).await {
        Some(p) => Json(p).into_response(),
        None => error_response_with(
            StatusCode::NOT_FOUND,
            "person_not_found",
            "no such person",
            None,
        ),
    }
}

pub async fn create_person(
    State((state, _)): State<AppState>,
    Json(body): Json<PersonCreate>,
) -> Response {
    if body.display_name.trim().is_empty() {
        return error_response_with(
            StatusCode::BAD_REQUEST,
            "display_name_required",
            "display_name is required",
            None,
        );
    }
    match state.gallery.create_person(body).await {
        Ok(p) => Json(p).into_response(),
        Err(e) => error_response_with(StatusCode::INTERNAL_SERVER_ERROR, "internal", &e, None),
    }
}

pub async fn archive_person(
    State((state, _)): State<AppState>,
    Path(id): Path<String>,
) -> Response {
    match state.gallery.archive_person(&id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response_with(StatusCode::INTERNAL_SERVER_ERROR, "internal", &e, None),
    }
}

pub async fn enroll_face(
    State((state, _)): State<AppState>,
    Path(person_id): Path<String>,
    mut multipart: Multipart,
) -> Response {
    let mut image_bytes: Option<Vec<u8>> = None;
    let mut bbox: Option<[i32; 4]> = None;
    let mut source_key: Option<String> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "image" => match field.bytes().await {
                Ok(b) => image_bytes = Some(b.to_vec()),
                Err(e) => {
                    return error_response_with(
                        StatusCode::BAD_REQUEST,
                        "read_field",
                        &format!("read_field: {e}"),
                        None,
                    );
                }
            },
            "bbox" => {
                if let Ok(s) = field.text().await {
                    if let Some(arr) = parse_bbox(&s) {
                        bbox = Some(arr);
                    }
                }
            }
            "source_key" => {
                source_key = field.text().await.ok();
            }
            _ => {}
        }
    }
    let image_bytes = match image_bytes {
        Some(b) => b,
        None => {
            return error_response_with(
                StatusCode::BAD_REQUEST,
                "file_required",
                "multipart `file` field is required",
                None,
            );
        }
    };
    if image_bytes.is_empty() {
        return error_response_with(
            StatusCode::BAD_REQUEST,
            "file_empty",
            "uploaded file is 0 bytes",
            None,
        );
    }
    if image_bytes.len() > state.cfg.upload_limit_image {
        return error_response_with(
            StatusCode::BAD_REQUEST,
            "file_too_large",
            "uploaded file exceeds image upload limit",
            None,
        );
    }
    match state
        .gallery
        .enroll_face(&person_id, &image_bytes, FaceEnroll { bbox, source_key })
        .await
    {
        Ok(face) => Json(face).into_response(),
        Err(e) => classify_enroll_err(&e),
    }
}

fn classify_enroll_err(e: &str) -> Response {
    // Stable contract — the snake_case codes are part of the public
    // API. Frontends grep for them.
    match e {
        "person_not_found" => error_response_with(
            StatusCode::NOT_FOUND,
            "person_not_found",
            "no such person",
            None,
        ),
        "decode_failed" => error_response_with(
            StatusCode::BAD_REQUEST,
            "decode_failed",
            "image decode failed (only PNG/PGM/PPM are supported)",
            None,
        ),
        "no_face" => error_response_with(
            StatusCode::BAD_REQUEST,
            "no_face",
            "no face detected in the image",
            None,
        ),
        "detector_build_failed" => error_response_with(
            StatusCode::INTERNAL_SERVER_ERROR,
            "detector_build_failed",
            "internal: failed to load Haar cascade",
            None,
        ),
        "embed_failed" => error_response_with(
            StatusCode::INTERNAL_SERVER_ERROR,
            "embed_failed",
            "internal: EmbedNet forward pass collapsed",
            None,
        ),
        _ => error_response_with(StatusCode::INTERNAL_SERVER_ERROR, "internal", e, None),
    }
}

pub async fn list_faces(
    State((state, _)): State<AppState>,
    Path(person_id): Path<String>,
) -> Response {
    let faces = state.gallery.list_faces(&person_id).await;
    Json(FaceList {
        count: faces.len(),
        faces,
    })
    .into_response()
}

#[derive(Debug, Serialize)]
pub struct FaceList {
    pub count: usize,
    pub faces: Vec<FaceMeta>,
}

pub async fn delete_face(State((state, _)): State<AppState>, Path(id): Path<String>) -> Response {
    match state.gallery.delete_face(&id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) if e == "face_not_found" => error_response_with(
            StatusCode::NOT_FOUND,
            "face_not_found",
            "no such face",
            None,
        ),
        Err(e) => error_response_with(StatusCode::INTERNAL_SERVER_ERROR, "internal", &e, None),
    }
}

#[derive(Debug, Serialize)]
pub struct IdentifyResult {
    pub face_count: usize,
    pub width: usize,
    pub height: usize,
    pub faces: Vec<crate::gallery::RankedFace>,
    pub gallery_identities: usize,
    pub gallery_embeddings: usize,
    pub threshold: f32,
}

pub async fn identify_image(
    State((state, _)): State<AppState>,
    mut multipart: Multipart,
) -> Response {
    let mut image_bytes: Option<Vec<u8>> = None;
    let mut top_k: usize = 3;
    let mut algo_name = "haar".to_string();
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "image" => match field.bytes().await {
                Ok(b) => image_bytes = Some(b.to_vec()),
                Err(e) => {
                    return error_response_with(
                        StatusCode::BAD_REQUEST,
                        "read_field",
                        &format!("read_field: {e}"),
                        None,
                    );
                }
            },
            "top_k" => {
                if let Ok(s) = field.text().await {
                    if let Ok(n) = s.parse::<usize>() {
                        top_k = n.clamp(1, 50);
                    }
                }
            }
            "algo" => {
                if let Ok(s) = field.text().await {
                    algo_name = s;
                }
            }
            _ => {}
        }
    }
    let bytes = match image_bytes {
        Some(b) => b,
        None => {
            return error_response_with(
                StatusCode::BAD_REQUEST,
                "file_required",
                "multipart `file` field is required",
                None,
            );
        }
    };
    if bytes.is_empty() {
        return error_response_with(
            StatusCode::BAD_REQUEST,
            "file_empty",
            "uploaded file is 0 bytes",
            None,
        );
    }
    if bytes.len() > state.cfg.upload_limit_image {
        return error_response_with(
            StatusCode::BAD_REQUEST,
            "file_too_large",
            "uploaded file exceeds image upload limit",
            None,
        );
    }
    let gray = match decode_gray(&bytes) {
        Some(g) => g,
        None => return classify_enroll_err("decode_failed"),
    };
    let rgb = decode_rgb(&bytes).unwrap_or_else(|| rsface::image::RgbImage::from_gray(&gray));
    let width = gray.width();
    let height = gray.height();
    let detector = match build_detector(&algo_name, &state.cfg.cascade_path) {
        Some(d) => d,
        None => {
            return error_response_with(
                StatusCode::BAD_REQUEST,
                "unknown_algo",
                "only `haar` is currently supported",
                None,
            );
        }
    };
    let faces = state.gallery.recognize(detector, &rgb, &gray, top_k).await;
    Json(IdentifyResult {
        face_count: faces.len(),
        width,
        height,
        faces,
        gallery_identities: state.gallery.identity_count().await,
        gallery_embeddings: state.gallery.face_count().await,
        threshold: state.gallery.cfg().threshold,
    })
    .into_response()
}

#[derive(Debug, Serialize)]
pub struct VerifyResult {
    pub label: String,
    pub matched: bool,
    pub similarity: f32,
    pub threshold: f32,
}

pub async fn verify_image(
    State((state, _)): State<AppState>,
    mut multipart: Multipart,
) -> Response {
    let mut image_bytes: Option<Vec<u8>> = None;
    let mut label: Option<String> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "image" => match field.bytes().await {
                Ok(b) => image_bytes = Some(b.to_vec()),
                Err(e) => {
                    return error_response_with(
                        StatusCode::BAD_REQUEST,
                        "read_field",
                        &format!("read_field: {e}"),
                        None,
                    );
                }
            },
            "label" => {
                label = field.text().await.ok();
            }
            _ => {}
        }
    }
    let bytes = match image_bytes {
        Some(b) => b,
        None => {
            return error_response_with(
                StatusCode::BAD_REQUEST,
                "file_required",
                "multipart `file` field is required",
                None,
            );
        }
    };
    let label = match label {
        Some(l) if !l.is_empty() => l,
        _ => {
            return error_response_with(
                StatusCode::BAD_REQUEST,
                "label_required",
                "multipart `label` field is required",
                None,
            );
        }
    };
    if bytes.is_empty() {
        return error_response_with(
            StatusCode::BAD_REQUEST,
            "file_empty",
            "uploaded file is 0 bytes",
            None,
        );
    }
    let gray = match decode_gray(&bytes) {
        Some(g) => g,
        None => return classify_enroll_err("decode_failed"),
    };
    let threshold = state.gallery.cfg().threshold;
    match state.gallery.verify(&gray, &label).await {
        Some(sim) => Json(VerifyResult {
            label,
            matched: sim >= threshold,
            similarity: sim,
            threshold,
        })
        .into_response(),
        None => error_response_with(
            StatusCode::NOT_FOUND,
            "label_not_enrolled",
            "no enrolled embeddings for that label",
            None,
        ),
    }
}

// --- helpers ---

fn parse_bbox(s: &str) -> Option<[i32; 4]> {
    let parts: Vec<i32> = s
        .split([',', ' ', ';'])
        .filter_map(|t| t.parse::<i32>().ok())
        .collect();
    if parts.len() == 4 {
        Some([parts[0], parts[1], parts[2], parts[3]])
    } else {
        None
    }
}

fn decode_gray(bytes: &[u8]) -> Option<rsface::image::GrayImage> {
    use std::io::Cursor;
    let mut cur = Cursor::new(bytes);
    if let Ok(g) = decode_to_gray(&mut cur) {
        return Some(g);
    }
    let mut cur = Cursor::new(bytes);
    if let Ok(g) = read_pgm(&mut cur) {
        return Some(g);
    }
    let mut cur = Cursor::new(bytes);
    rsface::image::codec::read_ppm(&mut cur)
        .ok()
        .map(|img| img.to_gray())
}

/// Decode an uploaded image to RGB, trying PNG/JPEG and falling back to
/// reconstructing colour from a grayscale decode. Liveness needs RGB.
fn decode_rgb(bytes: &[u8]) -> Option<rsface::image::RgbImage> {
    use std::io::Cursor;
    let mut cur = Cursor::new(bytes);
    if let Ok(rgb) = rsface::image::png::decode_to_rgb(&mut cur) {
        return Some(rgb);
    }
    if let Ok(rgb) = rsface::image::jpeg::decode_jpeg_rgb(bytes) {
        return Some(rgb);
    }
    decode_gray(bytes).map(|g| rsface::image::RgbImage::from_gray(&g))
}

fn build_detector(algo: &str, cascade_path: &std::path::Path) -> Option<DetectorKind> {
    if algo != "haar" {
        return None;
    }
    let cascade = rsface::haar::Cascade::load(cascade_path).ok()?;
    let cfg = rsface::detector::DetectorConfig::default();
    let detector = rsface::detector::Detector::new(cascade, cfg);
    Some(DetectorKind::Haar(detector))
}
