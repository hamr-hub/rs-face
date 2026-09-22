//! Optional silent face-anti-spoofing wired into `/identify`.
//!
//! The platform stays zero-dependency by default: this module only loads a
//! real MiniFASNet graph when the platform `liveness` Cargo feature is on
//! (which pulls in rs-face's pure-Rust `tract-backend`). With the feature off,
//! [`Liveness::open`] returns `None` and every check is skipped, so the JSON
//! API simply omits the liveness field.
//!
//! Enable at runtime with `RSFACE_LIVENESS_ENABLED=true` plus
//! `RSFACE_LIVENESS_MODELS_DIR` pointing at the two MiniFASNet ONNX graphs
//! fetched by `tools/fetch_models.sh`.

use crate::config::Config;
use rsface::face::Detection;
use rsface::image::RgbImage;
use serde::Serialize;

/// Serializable per-face liveness verdict returned by `/identify`.
#[derive(Clone, Debug, Serialize)]
pub struct LivenessVerdict {
    pub is_real: bool,
    pub real_score: f32,
    pub label: String,
}

/// Handle to an optional liveness backend.
///
/// The feature-gated build stores a real detector; the zero-dep build is a
/// unit type that never produces a verdict.
#[derive(Clone)]
pub struct Liveness {
    #[cfg(feature = "liveness")]
    // rs-face's `Inference` sessions are `Send` but deliberately not `Sync`.
    // The platform shares one backend across HTTP workers via JobRegistry, so
    // serialize access behind a Mutex (a check is ~9 ms; contention is minor).
    detector: std::sync::Arc<std::sync::Mutex<rsface::liveness_detector::LivenessDetector>>,
}

impl Liveness {
    /// Build the backend from configuration, returning `None` when liveness
    /// is disabled, the feature is not compiled in, or the models fail to load
    /// (a load failure is logged and never prevents the server from starting).
    pub fn open(cfg: &Config) -> Option<Self> {
        if !cfg.liveness_enabled {
            return None;
        }
        Self::open_models(cfg)
    }

    /// Run the check on one detected face; `None` signals "no verdict".
    pub fn check(&self, rgb: &RgbImage, det: &Detection) -> Option<LivenessVerdict> {
        self.run_check(rgb, det)
    }

    #[cfg(feature = "liveness")]
    fn open_models(cfg: &Config) -> Option<Self> {
        use rsface::liveness::LivenessConfig;
        use rsface::liveness_detector::LivenessDetector;
        use rsface::models::{LIVENESS_MINIFASNET_V1SE, LIVENESS_MINIFASNET_V2};
        use rsface::onnx::SessionConfig;

        let dir = &cfg.liveness_models_dir;
        let path_v2 = dir.join(LIVENESS_MINIFASNET_V2.file_name);
        let path_v1se = dir.join(LIVENESS_MINIFASNET_V1SE.file_name);
        let liveness_cfg = LivenessConfig {
            min_real_score: cfg.liveness_min_real_score,
        };
        match LivenessDetector::open(
            &path_v2,
            Some(&LIVENESS_MINIFASNET_V2),
            &path_v1se,
            Some(&LIVENESS_MINIFASNET_V1SE),
            &SessionConfig::default(),
            liveness_cfg,
        ) {
            Ok(detector) => {
                tracing::info!(
                    "[liveness] enabled: models_dir={} heads={}",
                    dir.display(),
                    detector.num_heads()
                );
                Some(Self {
                    detector: std::sync::Arc::new(std::sync::Mutex::new(detector)),
                })
            }
            Err(e) => {
                tracing::warn!(
                    "[liveness] enabled but models failed to load from {}: {e:?} (liveness will be unavailable)",
                    dir.display()
                );
                None
            }
        }
    }

    #[cfg(not(feature = "liveness"))]
    fn open_models(_cfg: &Config) -> Option<Self> {
        tracing::warn!(
            "[liveness] RSFACE_LIVENESS_ENABLED=true but the platform was built without the \
             `liveness` cargo feature — rebuild with `--features liveness` to enable it"
        );
        None
    }

    #[cfg(feature = "liveness")]
    fn run_check(&self, rgb: &RgbImage, det: &Detection) -> Option<LivenessVerdict> {
        let detector = self.detector.lock().ok()?;
        let outcome = detector.check(rgb, det).ok()?;
        Some(LivenessVerdict {
            is_real: outcome.is_real,
            real_score: outcome.real_score,
            label: outcome.label().to_string(),
        })
    }

    #[cfg(not(feature = "liveness"))]
    fn run_check(&self, _rgb: &RgbImage, _det: &Detection) -> Option<LivenessVerdict> {
        None
    }
}
