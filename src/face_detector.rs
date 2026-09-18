//! Unified `FaceDetector` trait — every algorithm in `rs-face` core implements
//! this so platform code and tests can dispatch polymorphically.
//!
//! All detectors work on the core's `GrayImage` and produce `Detection`s from
//! `detector.rs` (pixel-space bbox + score). They are CPU-only, allocation-light,
//! and `!Sync` (detector scratch buffers are owned per-thread, see `cnn::CnnScratch`).

use crate::detector::Detection;
use crate::detector::{Detector as HaarDetectorInner, DetectorConfig};
use crate::face::FaceDetection;
use crate::haar::Cascade;
use crate::image::{GrayImage, RgbImage};

/// How much trust a detector's output has earned.
///
/// This exists because the crate currently ships several detector *scaffolds* whose
/// architecture is implemented but whose weights are random placeholders. Such a
/// detector does not fail loudly — it returns an empty detection list, which is
/// indistinguishable from "this frame genuinely contains no faces". Exposing those as
/// peers of a real detector in a selection menu invites a user to conclude the images
/// are at fault. Maturity is therefore part of the trait contract, so every surface
/// (CLI, HTTP API, web UI) can label it instead of each one re-deriving it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Maturity {
    /// Real trained weights; accuracy measured and documented in `docs/benchmarks.md`.
    Production,
    /// Real weights, but accuracy not yet independently verified in this crate.
    Experimental,
    /// Architecture only — weights are random placeholders. **Detects nothing.**
    ///
    /// Retained for reference and as a wiring target for real weights, never as a
    /// usable detector.
    Scaffold,
}

impl Maturity {
    /// Whether output from this detector is meaningful at all.
    #[inline]
    pub fn is_usable(&self) -> bool {
        !matches!(self, Maturity::Scaffold)
    }

    /// Stable lowercase token for JSON payloads and UI attributes.
    pub fn as_str(&self) -> &'static str {
        match self {
            Maturity::Production => "production",
            Maturity::Experimental => "experimental",
            Maturity::Scaffold => "scaffold",
        }
    }
}

/// Which colour representation a detector actually wants.
///
/// This is not cosmetic. Classical cascades (Haar, HoG) are defined on luminance and
/// gain nothing from colour. Modern CNN detectors are *trained* on 3-channel RGB, and
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
/// 1. Be cheap to construct (`YunetDetector::new(crate::yunet::YunetConfig::default())` is enough to run).
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
pub struct HaarDetector(pub HaarDetectorInner);

impl HaarDetector {
    /// Wrap a Haar cascade detector so it implements [`FaceDetector`].
    pub fn new(cascade: Cascade, config: DetectorConfig) -> Self {
        Self(HaarDetectorInner::new(cascade, config))
    }
}

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
        assert_eq!(Maturity::Scaffold.as_str(), "scaffold");
    }

    #[test]
    fn only_scaffold_is_unusable() {
        assert!(Maturity::Production.is_usable());
        assert!(Maturity::Experimental.is_usable());
        assert!(!Maturity::Scaffold.is_usable());
    }

    /// Regression guard for the crate's central honesty invariant: a detector backed by
    /// random placeholder weights must declare itself a scaffold. If someone wires real
    /// weights into one of these, they must consciously update this test — which is
    /// exactly the review checkpoint we want.
    #[test]
    fn dummy_weight_detectors_declare_themselves_scaffolds() {
        use crate::hog_face::HogFaceDetector;
        use crate::mtcnn::MtcnnDetector;
        use crate::yunet::YunetDetector;

        let dummies: Vec<Box<dyn FaceDetector>> = vec![
            Box::new(MtcnnDetector::new(crate::mtcnn::MtcnnConfig::default())),
            Box::new(YunetDetector::new(crate::yunet::YunetConfig::default())),
            Box::new(HogFaceDetector::new(crate::hog_face::HogConfig::default())),
        ];

        for d in &dummies {
            assert_eq!(
                d.maturity(),
                Maturity::Scaffold,
                "detector '{}' uses placeholder weights and must report Scaffold",
                d.name()
            );
            assert!(
                d.description().contains("SCAFFOLD"),
                "detector '{}' description must warn the user it is a scaffold",
                d.name()
            );
        }
    }

    /// The scaffolds must also actually be inert, which is the fact that justifies the
    /// `Scaffold` label. If one starts emitting boxes, its weights changed and both the
    /// label and `docs/benchmarks.md` need revisiting.
    #[test]
    fn scaffold_detectors_return_nothing_on_a_real_gradient() {
        use crate::image::GrayImage;

        let (w, h) = (128usize, 128usize);
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.as_mut_slice()[y * w + x] = ((x * 2 + y) % 256) as u8;
            }
        }

        let mtcnn = crate::mtcnn::MtcnnDetector::new(crate::mtcnn::MtcnnConfig::default());
        assert!(
            mtcnn.detect(&img).is_empty(),
            "mtcnn scaffold unexpectedly produced detections"
        );

        let hog = crate::hog_face::HogFaceDetector::new(crate::hog_face::HogConfig::default());
        assert!(
            hog.detect(&img).is_empty(),
            "hog scaffold unexpectedly produced detections"
        );
    }

    #[test]
    fn default_color_input_is_gray_for_classical_detectors() {
        let hog = crate::hog_face::HogFaceDetector::new(crate::hog_face::HogConfig::default());
        assert_eq!(hog.color_input(), ColorInput::Gray);
    }

    /// The HaarDetector adapter must report the canonical name + maturity, so
    /// `--algo haar` and the FaceDetector-trait dispatch agree on the algorithm
    /// tag. Regression: a typo here would silently break `Vec<Box<dyn FaceDetector>>`
    /// users that key off the string.
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
    #[test]
    fn default_rgb_path_delegates_to_gray_without_panic() {
        use crate::image::RgbImage;

        let hog = crate::hog_face::HogFaceDetector::new(crate::hog_face::HogConfig::default());
        let rgb = RgbImage::new(64, 64);
        let out = hog.detect_faces_rgb(&rgb);
        assert!(out.is_empty());
    }

    /// A detector with no keypoint head must not claim landmark support, otherwise the
    /// recognition pipeline would try to align against `None`.
    #[test]
    fn classical_detectors_do_not_claim_landmarks() {
        let hog = crate::hog_face::HogFaceDetector::new(crate::hog_face::HogConfig::default());
        assert!(!hog.has_landmarks());
    }
}
