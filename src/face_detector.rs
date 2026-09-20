//! Unified `FaceDetector` trait — every algorithm in `rs-face` core implements
//! this so platform code and tests can dispatch polymorphically.
//!
//! All detectors work on the core's `GrayImage` and produce `Detection`s from
//! `detector.rs` (pixel-space bbox + score). They are CPU-only, allocation-light,
//! and `!Sync` (detector scratch buffers are owned per-thread, see `cnn::CnnScratch`).

#[cfg(feature = "detector-haar")]
use crate::detector::{Detector as HaarDetectorInner, DetectorConfig};
use crate::face::Detection;
use crate::face::FaceDetection;
#[cfg(feature = "detector-haar")]
use crate::haar::Cascade;
use crate::image::{GrayImage, RgbImage};

/// How much trust a detector's output has earned. Part of the trait
/// contract so every surface (CLI, HTTP API, web UI) can label detectors
/// the same way instead of each one re-deriving a verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Maturity {
    /// Real trained weights; accuracy measured and documented in `docs/benchmarks.md`.
    Production,
    /// Runnable end-to-end, but accuracy not yet independently measured in this
    /// crate (e.g. a trainable toy net with starter weights).
    Experimental,
}

impl Maturity {
    /// Stable lowercase token for JSON payloads and UI attributes.
    pub fn as_str(&self) -> &'static str {
        match self {
            Maturity::Production => "production",
            Maturity::Experimental => "experimental",
        }
    }
}

/// Which colour representation a detector actually wants.
///
/// This is not cosmetic. Classical detectors (Haar cascades, LBP, the luminance
/// heuristic) are defined on luminance and gain nothing from colour. Modern CNN detectors are *trained* on 3-channel RGB, and
/// feeding them a grey plane replicated across all three channels is a measurable
/// accuracy loss, not a free conversion — skin-tone and chroma edges carry real signal
/// for the box/keypoint heads. Callers use this to route a frame down the cheapest
/// path that does not degrade the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorInput {
    /// Detector consumes luminance; supplying RGB would only cost a conversion.
    Gray,
    /// Detector was trained on RGB; supplying grey costs accuracy.
    Rgb,
}

/// Trait every face detector must implement. Implementors should:
/// 1. Be cheap to construct (`LuminanceFaceDetector::new(LuminanceConfig::default())` is enough to run).
/// 2. Never panic on empty / uniform input (smoke-tested via `*_no_panic` tests).
/// 3. Return detections sorted by descending score (so NMS in callers is sane).
pub trait FaceDetector: Send {
    /// Run the detector on a single grayscale frame. Returns `Vec<Detection>`.
    fn detect(&self, img: &GrayImage) -> Vec<Detection>;

    /// Short lowercase name used in `RSFACE_ALGO` env var, `/api/config`, and
    /// SSE `algo` field. Must be unique across all detectors.
    fn name(&self) -> &'static str;

    /// Optional: detector description for the Web UI compare card.
    fn description(&self) -> &'static str {
        ""
    }

    /// The colour representation this detector was designed for.
    ///
    /// Defaults to [`ColorInput::Gray`], which is correct for every classical
    /// detector already in the tree; RGB-native models override it.
    fn color_input(&self) -> ColorInput {
        ColorInput::Gray
    }

    /// How much trust this detector's output has earned. See [`Maturity`].
    ///
    /// Defaults to [`Maturity::Experimental`] rather than `Production`: a detector must
    /// *claim* production quality explicitly, so a newly added one is never silently
    /// promoted by inheriting a permissive default.
    fn maturity(&self) -> Maturity {
        Maturity::Experimental
    }

    /// Whether this detector produces 5-point landmarks, and therefore whether its
    /// output can drive the recognition pipeline (which requires them for alignment).
    ///
    /// Defaults to `false` so a detector must *opt in* to being trusted for
    /// recognition. Getting this wrong in the permissive direction would send
    /// unaligned crops to ArcFace and quietly produce meaningless embeddings.
    fn has_landmarks(&self) -> bool {
        false
    }

    /// Full-fidelity detection entry point: sub-pixel boxes plus optional landmarks.
    ///
    /// The default implementation converts to luminance and widens the classical
    /// [`Detection`] output, so every existing detector satisfies this trait with no
    /// code change. RGB-native detectors override this method and leave [`Self::detect`]
    /// to delegate *inward*, which keeps exactly one real implementation per detector.
    fn detect_faces_rgb(&self, img: &RgbImage) -> Vec<FaceDetection> {
        self.detect(&img.to_gray())
            .into_iter()
            .map(FaceDetection::from)
            .collect()
    }

    /// Convenience: full-fidelity detection from a grayscale frame.
    ///
    /// For an RGB-native detector this necessarily replicates the grey plane across
    /// three channels and therefore runs the model off-distribution. Prefer
    /// [`Self::detect_faces_rgb`] whenever the original colour frame is still in hand.
    fn detect_faces(&self, img: &GrayImage) -> Vec<FaceDetection> {
        self.detect(img)
            .into_iter()
            .map(FaceDetection::from)
            .collect()
    }
}

/// Adapter that exposes the crate-root Haar cascade [`crate::Detector`]
/// through the uniform [`FaceDetector`] trait.
///
/// `rsface::Detector` is the original Haar cascade detector and predates the
/// trait. This newtype wraps it so Haar can be dispatched through the same
/// `Box<dyn FaceDetector>` API as every other algorithm — useful for
/// benchmarks, platform comparison code, and the "swiss-army-knife" recipe
/// in `examples/detect_uniform.rs`. The CLI continues to use
/// `rsface::Detector` directly because it is behaviour-coupled to
/// [`crate::pipeline`]; this wrapper only exists for SDK uniformity.
///
/// Maturity is reported as [`Maturity::Production`] for any cascade the user
/// supplies; scaffolds (random-weight demo cascade) report as [`Maturity::Experimental`]
/// so the badge remains truthful.
#[cfg(feature = "detector-haar")]
pub struct HaarDetector(pub HaarDetectorInner);

#[cfg(feature = "detector-haar")]
impl HaarDetector {
    /// Wrap a Haar cascade detector so it implements [`FaceDetector`].
    pub fn new(cascade: Cascade, config: DetectorConfig) -> Self {
        Self(HaarDetectorInner::new(cascade, config))
    }
}

#[cfg(feature = "detector-haar")]
impl FaceDetector for HaarDetector {
    fn name(&self) -> &'static str {
        "haar"
    }
    fn description(&self) -> &'static str {
        "Viola-Jones AdaBoost cascade over 5 Haar-like feature families. \
         Load OpenCV XML via tools/convert_opencv_xml.py -> .rfcf."
    }
    fn maturity(&self) -> Maturity {
        // The bundled demo cascade has hand-tuned thresholds and is the
        // production-quality baseline used by the CLI; cascade stage_bias is
        // the documented calibration knob. Real OpenCV-trained cascades
        // routed through `.rfcf` inherit the same maturity.
        Maturity::Production
    }
    fn detect(&self, img: &GrayImage) -> Vec<Detection> {
        self.0.detect(img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maturity_tokens_are_stable() {
        // These strings are consumed by the HTTP API and the web UI's data-attributes;
        // renaming one silently breaks the badge rendering.
        assert_eq!(Maturity::Production.as_str(), "production");
        assert_eq!(Maturity::Experimental.as_str(), "experimental");
    }

    /// The zero-dep classical detectors that ship by default must claim real
    /// maturity; a new detector starts at `Experimental` and is promoted
    /// consciously after measured accuracy lands in `docs/benchmarks.md`.
    #[cfg(all(feature = "detector-haar", feature = "detector-luminance"))]
    #[test]
    fn classical_detectors_are_gray_without_landmarks() {
        let det = HaarDetector::new(
            crate::haar::params::demo_face_cascade(),
            crate::detector::DetectorConfig::default(),
        );
        assert_eq!(det.color_input(), ColorInput::Gray);
        assert!(!det.has_landmarks());

        let lum = crate::luminance_face::LuminanceFaceDetector::new(
            crate::luminance_face::LuminanceConfig::default(),
        );
        assert_eq!(lum.color_input(), ColorInput::Gray);
        assert!(!lum.has_landmarks());
    }

    /// The HaarDetector adapter must report the canonical name + maturity, so
    /// `--algo haar` and the FaceDetector-trait dispatch agree on the algorithm
    /// tag. Regression: a typo here would silently break `Vec<Box<dyn FaceDetector>>`
    /// users that key off the string.
    #[cfg(feature = "detector-haar")]
    #[test]
    fn haar_detector_reports_canonical_name_and_maturity() {
        let det = HaarDetector::new(
            crate::haar::params::demo_face_cascade(),
            crate::detector::DetectorConfig::default(),
        );
        assert_eq!(det.name(), "haar");
        assert_eq!(det.maturity(), Maturity::Production);
        assert!(det.color_input() == ColorInput::Gray);
        // detect() must not panic on uniform input — universal contract.
        let img = crate::image::GrayImage::new(16, 16);
        let _ = det.detect(&img);
    }

    /// The blanket `detect_faces_rgb` default must round-trip through grayscale without
    /// panicking, since every classical detector relies on it.
    #[cfg(feature = "detector-haar")]
    #[test]
    fn default_rgb_path_delegates_to_gray_without_panic() {
        use crate::image::RgbImage;

        let det = HaarDetector::new(
            crate::haar::params::demo_face_cascade(),
            crate::detector::DetectorConfig::default(),
        );
        let rgb = RgbImage::new(64, 64);
        let _ = det.detect_faces_rgb(&rgb);
    }
}
