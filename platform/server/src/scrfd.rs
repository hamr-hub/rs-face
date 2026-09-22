//! SCRFD-10G (5-keypoint) face detector wired into the platform.
//!
//! SCRFD is the only detector we ship for the ArcFace pipeline because it
//! is the only one that emits the **5-point landmarks** the alignment step
//! needs to warp the crop onto the ArcFace canonical layout. The legacy
//! Haar / CNN / Luminance chain stops at the bounding box, so embedding a
//! bare crop is technically possible but produces an off-distribution
//! vector (cosine drops to chance).
//!
//! Like [`crate::arcface`], the session is wrapped in a `Mutex` because
//! `rsface::onnx::Inference` is `Send` but deliberately not `Sync`.
//! Throughput is dominated by the GPU forward pass, not lock contention.
//!
//! On failure the wrapper returns `Err`, not `None`: callers need to
//! distinguish "SCRFD didn't compile in" from "graph failed to load" to
//! decide whether to soft-fall back to the no-landmark path.

use crate::arcface::resolve_scrfd_path;
use crate::config::{ArcFaceBackend, Config};
use rsface::face::FaceDetection;
use rsface::image::RgbImage;
use rsface::models;
use rsface::onnx::{Backend, OnnxError, SessionConfig};
use rsface::scrfd::ScrfdConfig;
use rsface::scrfd_detector::ScrfdDetector;
use std::sync::{Arc, Mutex};

/// Loaded SCRFD detector. `None` when no backend compiled in / graphs missing.
#[derive(Clone)]
pub struct Scrfd {
    detector: Arc<Mutex<ScrfdDetector>>,
}

impl Scrfd {
    pub fn open(cfg: &Config) -> Result<Option<Self>, String> {
        if !cfg.arcface_enabled {
            return Ok(None);
        }
        if Backend::available().is_empty() {
            return Err(
                "SCRFD needs an ONNX backend (compile with rsface/tract-backend or rsface/ort-backend)"
                    .to_string(),
            );
        }
        let session_cfg = match cfg.arcface_backend {
            ArcFaceBackend::Auto => SessionConfig::default(),
            ArcFaceBackend::Ort => SessionConfig::default().with_backend(Backend::Ort),
            ArcFaceBackend::Tract => SessionConfig::default().with_backend(Backend::Tract),
        };
        let path = resolve_scrfd_path(cfg);
        let spec = models::find("scrfd_10g_kps").ok_or_else(|| {
            "SCRFD 10G spec missing from rsface::models::REGISTRY".to_string()
        })?;
        match ScrfdDetector::open(&path, Some(spec), &session_cfg, ScrfdConfig::default()) {
            Ok(det) => {
                tracing::info!(
                    "[scrfd] loaded: path={} kps={} backend={:?}",
                    path.display(),
                    det.has_keypoints(),
                    cfg.arcface_backend
                );
                if !det.has_keypoints() {
                    return Err(format!(
                        "SCRFD graph at {} has no keypoint head; ArcFace alignment is \
                         impossible without landmarks. Export with keypoints enabled.",
                        path.display()
                    ));
                }
                Ok(Some(Self {
                    detector: Arc::new(Mutex::new(det)),
                }))
            }
            Err(e) => Err(format!(
                "SCRFD failed to load ({}): {:?}",
                path.display(),
                e
            )),
        }
    }

    /// Detect every face in `rgb`. Returns detections with 5-point landmarks
    /// (the only kind the platform uses downstream). Errors are propagated
    /// rather than swallowed — an empty frame is `Ok(vec![])`, an inference
    /// failure is `Err`.
    pub fn detect(&self, rgb: &RgbImage) -> Result<Vec<FaceDetection>, OnnxError> {
        let det = self.detector.lock().expect("scrfd mutex poisoned");
        det.detect_rgb(rgb)
    }
}