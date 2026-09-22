//! Industrial face-recogniser wiring.
//!
//! Replace the in-tree zero-dep `EmbedNet` (random init, 128-d, no pretrained
//! weights) with the actual model the platform was built to ship:
//! **InsightFace `w600k_r50` ArcFace R50**, 512-d L2-normalised embeddings.
//!
//! The recognition pipeline that finally produces a usable similarity is
//! strictly longer than the legacy box→resize one:
//!
//! ```text
//! RgbImage
//!   └─▶ SCRFD-10G (det_10g.onnx)          : face box + 5 landmarks
//!         └─▶ crate::align::norm_crop      : 112×112 ArcFace canonical
//!               └─▶ ArcFace w600k_r50      : 512-d embedding
//!                     └─▶ L2-normalise     : cosine-ready
//! ```
//!
//! On its own an SCRFD box is not enough: the crop has to be warped onto the
//! canonical landmark layout, otherwise the embedding is computed off-distribution
//! and the cosine drops to chance. [`ArcFaceRecognizer::embed`] enforces that
//! by demanding [`Landmarks`] — there is no entry point that takes a bare
//! bounding box.
//!
//! ## Backend
//!
//! Selection goes through [`crate::config::ArcFaceBackend`]:
//! `auto` (default) prefers the fastest compiled backend (`ort` over `tract`).
//! The pure-Rust `tract` backend has no system dependency and runs on CPU;
//! `ort` requires `libonnxruntime.so` on the host (vendored via the GPU image's
//! `apt-get install` step). Both compile into the same [`rsface::onnx::Inference`]
//! trait, so the rest of the platform sees a single recogniser.

use crate::config::{ArcFaceBackend, Config};
use rsface::arcface_recognizer::ArcFaceRecognizer;
use rsface::face::Landmarks;
use rsface::image::RgbImage;
use rsface::models::{self, ARCFACE_W600K_R50};
use rsface::onnx::{Backend, SessionConfig};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Loaded ArcFace (InsightFace w600k_r50, 512-d). `None` when the graphs
/// could not be loaded — see [`ArcFace::open`] for the soft-fallback path.
#[derive(Clone)]
pub struct ArcFace {
    recognizer: Arc<Mutex<ArcFaceRecognizer>>,
    backend: ArcFaceBackend,
}

impl ArcFace {
    /// Try to build from `cfg`. `Ok(None)` is the normal "graphs missing /
    /// feature off / no backend compiled in" path; callers decide whether to
    /// soft-fall-back to EmbedNet or hard-fail.
    pub fn open(cfg: &Config) -> Result<Option<Self>, String> {
        if !cfg.arcface_enabled {
            tracing::info!("[arcface] disabled by RSFACE_ARCFACE_ENABLED=false");
            return Ok(None);
        }
        let backend = pick_backend(cfg)?;
        let path = match &cfg.arcface_weights {
            Some(p) => p.clone(),
            None => cfg.arcface_models_dir.join(ARCFACE_W600K_R50.file_name),
        };
        let spec = models::find("arcface_w600k_r50").ok_or_else(|| {
            "ArcFace recogniser spec missing from rsface::models::REGISTRY".to_string()
        })?;
        let session_cfg = match backend {
            ArcFaceBackend::Auto => SessionConfig::default(),
            ArcFaceBackend::Ort => SessionConfig::default().with_backend(Backend::Ort),
            ArcFaceBackend::Tract => SessionConfig::default().with_backend(Backend::Tract),
        };
        match ArcFaceRecognizer::open(&path, Some(spec), &session_cfg) {
            Ok(recog) => {
                tracing::info!(
                    "[arcface] loaded: path={} backend={} dim={}",
                    path.display(),
                    backend.as_str(),
                    recog.dim()
                );
                Ok(Some(Self {
                    recognizer: Arc::new(Mutex::new(recog)),
                    backend,
                }))
            }
            Err(e) => Err(format!(
                "ArcFace weights failed to load (path={}, backend={}): {e:?}",
                path.display(),
                backend.as_str()
            )),
        }
    }

    /// Backend actually used at open time (useful for diagnostics + smoke logs).
    pub fn backend(&self) -> ArcFaceBackend {
        self.backend
    }

    /// True iff the loaded graph is the standard 512-d ArcFace backbone.
    pub fn is_standard_dim(&self) -> bool {
        self.recognizer
            .lock()
            .ok()
            .is_some_and(|r| r.is_standard_arcface_dim())
    }

    /// Embed an aligned face from `(rgb, landmarks)`. Returns `None` when
    /// alignment produces a degenerate crop or the forward pass collapses.
    pub fn embed(&self, rgb: &RgbImage, lms: &Landmarks) -> Option<Vec<f32>> {
        let recog = self.recognizer.lock().ok()?;
        let emb = recog.embed(rgb, lms).ok()?;
        Some(emb.as_slice().to_vec())
    }
}

/// Resolve the user's `RSFACE_BACKEND` setting against the compiled-in
/// backend list and return a concrete enum. Logs and returns `Ok(None)` when
/// nothing usable is available — soft-fallback will keep EmbedNet working.
fn pick_backend(cfg: &Config) -> Result<ArcFaceBackend, String> {
    let want = cfg.arcface_backend;
    let available = Backend::available();
    if available.is_empty() {
        tracing::warn!(
            "[arcface] no ONNX backend compiled in (need --features rsface/tract-backend or \
             --features rsface/ort-backend); soft-falling back to EmbedNet"
        );
        // We still want to return a backend so the rest of the path can be
        // unit-tested without an ONNX runtime; we fail-open with a sentinel.
        return Err("no ONNX backend compiled in".to_string());
    }
    let resolved = match want {
        ArcFaceBackend::Auto => match available.first().copied() {
            Some(b) => match b {
                Backend::Ort => ArcFaceBackend::Ort,
                Backend::Tract => ArcFaceBackend::Tract,
            },
            None => return Err("no ONNX backend available".to_string()),
        },
        ArcFaceBackend::Ort => {
            if !available.contains(&Backend::Ort) {
                return Err(format!(
                    "RSFACE_ARCFACE_BACKEND=ort but ort-backend not compiled in \
                     (available: {:?})",
                    available
                ));
            }
            ArcFaceBackend::Ort
        }
        ArcFaceBackend::Tract => {
            if !available.contains(&Backend::Tract) {
                return Err(format!(
                    "RSFACE_ARCFACE_BACKEND=tract but tract-backend not compiled in \
                     (available: {:?})",
                    available
                ));
            }
            ArcFaceBackend::Tract
        }
    };
    Ok(resolved)
}

/// Resolve `cfg.scrfd_weights` or fall back to `<models_dir>/det_10g.onnx`.
pub(crate) fn resolve_scrfd_path(cfg: &Config) -> PathBuf {
    match &cfg.scrfd_weights {
        Some(p) => p.clone(),
        None => cfg.arcface_models_dir.join("det_10g.onnx"),
    }
}

/// Resolve `cfg.arcface_weights` or fall back to `<models_dir>/w600k_r50.onnx`.
pub(crate) fn resolve_arcface_path(cfg: &Config) -> PathBuf {
    match &cfg.arcface_weights {
        Some(p) => p.clone(),
        None => cfg.arcface_models_dir.join("w600k_r50.onnx"),
    }
}
