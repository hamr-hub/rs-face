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
    /// Native-crop quality measurements used to collect replay signals
    /// (focus/exposure/clipping/high-frequency) for threshold calibration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<QualitySnapshot>,
}

/// Serializable subset of the core quality report returned per face.
#[derive(Clone, Debug, Serialize)]
pub struct QualitySnapshot {
    pub width: usize,
    pub height: usize,
    pub sharpness: f32,
    pub mean_brightness: f32,
    pub clipped_ratio: f32,
    pub high_freq_ratio: f32,
}

impl From<&rsface::quality::QualityReport> for QualitySnapshot {
    fn from(q: &rsface::quality::QualityReport) -> Self {
        Self {
            width: q.width,
            height: q.height,
            sharpness: q.sharpness,
            mean_brightness: q.mean_brightness,
            clipped_ratio: q.clipped_ratio,
            high_freq_ratio: q.high_freq_ratio,
        }
    }
}

/// Decide whether liveness enforcement blocks a detection.
///
/// The gate is fail-closed: when enforcement is on and a backend is loaded,
/// only a confirmed real verdict passes. A missing verdict (no face box,
/// inference error or poisoned lock) blocks just like an explicit spoof
/// verdict, so an attacker cannot trigger an error to slip through. When
/// enforcement is off or no backend is loaded, nothing is blocked.
pub(crate) fn enforce_blocks(enforce: bool, has_backend: bool, is_real: Option<bool>) -> bool {
    enforce && has_backend && !matches!(is_real, Some(true))
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
        use rsface::quality::QualityConfig;

        let dir = &cfg.liveness_models_dir;
        let path_v2 = dir.join(LIVENESS_MINIFASNET_V2.file_name);
        let path_v1se = dir.join(LIVENESS_MINIFASNET_V1SE.file_name);
        let liveness_cfg = LivenessConfig {
            min_real_score: cfg.liveness_min_real_score,
        };
        let quality_cfg = if cfg.liveness_quality_gate {
            QualityConfig::strict()
        } else {
            QualityConfig::default()
        };
        match LivenessDetector::open(
            &path_v2,
            Some(&LIVENESS_MINIFASNET_V2),
            &path_v1se,
            Some(&LIVENESS_MINIFASNET_V1SE),
            &SessionConfig::default(),
            liveness_cfg,
            quality_cfg,
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
            quality: outcome.quality.as_ref().map(QualitySnapshot::from),
        })
    }

    #[cfg(not(feature = "liveness"))]
    fn run_check(&self, _rgb: &RgbImage, _det: &Detection) -> Option<LivenessVerdict> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{enforce_blocks, QualitySnapshot};

    #[test]
    fn real_verdict_passes_under_enforcement() {
        assert!(!enforce_blocks(true, true, Some(true)));
    }

    #[test]
    fn explicit_spoof_is_blocked() {
        assert!(enforce_blocks(true, true, Some(false)));
    }

    #[test]
    fn missing_verdict_is_fail_closed() {
        // Backend loaded but the check errored / produced no face box.
        assert!(enforce_blocks(true, true, None));
    }

    #[test]
    fn no_enforcement_or_no_backend_never_blocks() {
        assert!(!enforce_blocks(false, true, Some(false)));
        assert!(!enforce_blocks(true, false, Some(false)));
        assert!(!enforce_blocks(true, false, None));
    }

    #[test]
    fn quality_snapshot_serialises_the_calibration_contract() {
        use rsface::image::GrayImage;
        use rsface::quality::{assess, QualityConfig};

        // A small textured gray crop so every field takes a real value.
        let mut img = GrayImage::from_vec(vec![120u8; 40 * 40], 40, 40);
        for y in (1..39).step_by(2) {
            for x in (1..39).step_by(2) {
                img.as_mut_slice()[y * 40 + x] = 200;
            }
        }
        let report = assess(&img, &QualityConfig::default());
        let snapshot = QualitySnapshot::from(&report);
        let value = serde_json::to_value(&snapshot).expect("serialise");

        // Lock the field names downstream calibration tooling depends on.
        for field in [
            "width",
            "height",
            "sharpness",
            "mean_brightness",
            "clipped_ratio",
            "high_freq_ratio",
        ] {
            assert!(value.get(field).is_some(), "missing field `{field}`");
        }
        assert_eq!(value["width"], 40);
        assert_eq!(value["height"], 40);
        assert_eq!(value["sharpness"], snapshot.sharpness);
        assert!(value["high_freq_ratio"].as_f64().unwrap() > 0.0);
    }
}
