//! Unified [`FaceRecognizer`] trait — the recognition-side counterpart of
//! [`crate::face_detector::FaceDetector`].
//!
//! The three zero-dep gallery recognisers — LBPH (the `LbphRecognizer` type
//! in the `lbph` module), eigenfaces (`EigenfaceRecognizer` in `eigenface`)
//! and Fisherfaces (`FisherfaceRecognizer` in `fisherface`) — all consume [`GrayImage`]
//! crops, rank an enrolled gallery by a distance, gate the winner with a
//! threshold plus a runner-up margin, and report the same four outcomes.
//! That shared surface is encoded here so benchmark / platform / CLI code
//! can dispatch any of them through `&dyn FaceRecognizer`.
//!
//! Distances are **not** comparable across implementations: LBPH emits a
//! χ² distance over LBP histograms (default threshold 16.7), the subspace
//! recognisers emit Euclidean distance in coefficient space (defaults 6.3
//! / 3.0). Always use each recogniser's own configured threshold; the
//! [`Recognition`] enum carries the winner's raw distance for logging.
//!
//! ArcFace (the ONNX `arcface_recognizer` module) deliberately does NOT
//! implement this trait: it needs RGB + an ONNX runtime and scores *cosine
//! similarity* (higher is better) via [`crate::embedding::MatchOutcome`],
//! so folding it into this distance-based grey-crop trait would hide real
//! semantics. A grey-crop adapter may be added once an aligned-grey path
//! exists.
//!
//! ```
//! # #![cfg(feature = "recognizer-lbph")]
//! use rsface::image::GrayImage;
//! use rsface::lbph::{LbphConfig, LbphRecognizer};
//! use rsface::recognizer::{FaceRecognizer, IncrementalRecognizer};
//!
//! fn signed_off(rec: &dyn FaceRecognizer) -> bool {
//!     !rec.is_empty() && rec.name() == "lbph"
//! }
//!
//! let mut rec = LbphRecognizer::new(LbphConfig::default());
//! let crop = GrayImage::new(32, 32);
//! rec.enroll("alice", &crop);
//! assert!(signed_off(&rec));
//! ```

use crate::image::GrayImage;

/// Outcome of a gallery query, shared by every zero-dep recogniser.
///
/// This is the distance-valued analogue of
/// [`crate::embedding::MatchOutcome`] (the cosine-similarity result used by
/// ArcFace); the two stay separate so "smaller wins" and "larger wins"
/// scores can never be confused.
#[derive(Clone, Debug, PartialEq)]
pub enum Recognition {
    /// Best identity is within the recogniser's configured distance
    /// threshold (and satisfies the runner-up margin policy).
    Match {
        /// Gallery label of the winner.
        label: String,
        /// Winner's raw distance (χ² for LBPH, Euclidean for subspace).
        distance: f32,
        /// Distance gap to the runner-up (`f32::INFINITY` with one identity).
        margin: f32,
    },
    /// Nearest identity was farther than the threshold. The candidate is
    /// reported so callers can log near-misses and retune instead of only
    /// getting "no".
    BelowThreshold {
        /// The nearest-but-rejected candidate, if the gallery was non-empty.
        best: Option<(String, f32)>,
    },
    /// Within threshold but too close to another identity to call safely.
    Ambiguous {
        /// Nearest label.
        first: String,
        /// Runner-up label.
        second: String,
        /// Distance gap that proved too small.
        margin: f32,
    },
    /// Gallery empty / model not trained.
    NoCandidates,
}

impl Recognition {
    /// Label of the accepted identity, or `None` for every non-match outcome.
    pub fn label(&self) -> Option<&str> {
        match self {
            Recognition::Match { label, .. } => Some(label),
            _ => None,
        }
    }

    /// `true` only for [`Recognition::Match`].
    pub fn is_match(&self) -> bool {
        matches!(self, Recognition::Match { .. })
    }

    /// Raw distance of the reported candidate: the winner's distance for
    /// `Match`, the nearest candidate's for `BelowThreshold`, and `None`
    /// for `Ambiguous`/`NoCandidates`.
    pub fn distance(&self) -> Option<f32> {
        match self {
            Recognition::Match { distance, .. } => Some(*distance),
            Recognition::BelowThreshold { best: Some((_, d)) } => Some(*d),
            _ => None,
        }
    }
}

/// Uniform query surface for the three zero-dep gallery recognisers.
///
/// Implementors honour the universal contract from
/// [`crate::face_detector::FaceDetector`]: every method is panic-free on
/// empty / uniform input and an empty gallery yields
/// [`Recognition::NoCandidates`] rather than a panic.
pub trait FaceRecognizer: Send + Sync {
    /// Stable lowercase id (`"lbph"` / `"eigenface"` / `"fisherface"`),
    /// used by the CLI and platform dispatch exactly like
    /// [`crate::face_detector::FaceDetector::name`].
    fn name(&self) -> &'static str;

    /// Full gallery query with threshold + margin policy.
    fn identify_crop(&self, crop: &GrayImage) -> Recognition;

    /// Every enrolled identity with its distance, ascending (nearest
    /// first). The two recogniser-specific `rank` methods accept
    /// precomputed descriptors/coefficients; this is the grey-crop form.
    fn rank_crop(&self, crop: &GrayImage) -> Vec<(String, f32)>;

    /// Distance of `crop` to the gallery of `label`, or `None` if that
    /// label is not enrolled. Does NOT apply the accept threshold —
    /// callers gate the returned distance themselves.
    fn verify(&self, label: &str, crop: &GrayImage) -> Option<f32>;

    /// Number of distinct enrolled identities.
    fn len(&self) -> usize;

    /// Total crops stored across every identity (a gallery may hold
    /// multiple shots per person).
    fn crop_count(&self) -> usize;

    /// `true` when no identity is enrolled.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Recognisers whose gallery can change after construction.
///
/// LBPH supports incremental enrolment (descriptors are cheap to append).
/// The subspace recognisers do not: adding a person changes the PCA/LDA
/// basis, so they stay train-once values and are retrained via `train`.
pub trait IncrementalRecognizer: FaceRecognizer {
    /// Add one shot of `label` to the gallery (new label = new identity).
    ///
    /// Takes `String` (rather than `impl Into<String>`) so the trait stays
    /// dyn-compatible for platform/CLI dispatch.
    fn enroll(&mut self, label: String, crop: &GrayImage);

    /// Remove every shot of `label`; returns `true` if something was removed.
    fn remove(&mut self, label: &str) -> bool;
}

// The `FaceRecognizer` / `IncrementalRecognizer` impls live next to each
// algorithm (src/lbph.rs, src/eigenface.rs, src/fisherface.rs) so this trait
// module does not force every recogniser into every build.

#[cfg(all(test, feature = "recognizer-lbph"))]
mod tests {
    use super::*;
    use crate::lbph::{LbphConfig, LbphRecognizer};

    fn fake_face(seed: u8) -> GrayImage {
        // Structured per-seed pattern so different seeds land at different
        // histogram signatures; uniform grey would be a degenerate descriptor.
        let mut img = GrayImage::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                img[(x, y)] = ((x as u16 + y as u16 + seed as u16 * 7) % 200) as u8 + 40;
            }
        }
        img
    }

    #[test]
    fn trait_dispatch_reports_names_and_gallery_size() {
        let mut lbph = LbphRecognizer::new(LbphConfig::default());
        lbph.enroll("alice", &fake_face(1));
        lbph.enroll("alice", &fake_face(2));
        lbph.enroll("bob", &fake_face(9));

        let recs: Vec<Box<dyn FaceRecognizer>> = vec![Box::new(lbph)];
        let rec = &recs[0];
        assert_eq!(rec.name(), "lbph");
        assert_eq!(rec.len(), 2);
        assert_eq!(rec.crop_count(), 3);
        assert!(!rec.is_empty());
    }

    #[test]
    fn incremental_trait_enrols_and_removes() {
        fn enroll(rec: &mut dyn IncrementalRecognizer, label: &str, crop: &GrayImage) {
            rec.enroll(label.to_string(), crop);
        }
        let mut lbph = LbphRecognizer::new(LbphConfig::default());
        enroll(&mut lbph, "alice", &fake_face(1));
        assert_eq!(lbph.len(), 1);
        assert!(lbph.remove("alice"));
        assert!(lbph.is_empty());
        assert!(!lbph.remove("alice"));
    }

    #[test]
    fn empty_gallery_is_no_candidates_through_the_trait() {
        let lbph = LbphRecognizer::new(LbphConfig::default());
        let rec: &dyn FaceRecognizer = &lbph;
        assert_eq!(rec.identify_crop(&fake_face(3)), Recognition::NoCandidates);
        assert!(rec.rank_crop(&fake_face(3)).is_empty());
        assert_eq!(rec.verify("alice", &fake_face(3)), None);
    }

    #[test]
    fn recognition_helpers_extract_label_and_distance() {
        let m = Recognition::Match {
            label: "alice".into(),
            distance: 4.0,
            margin: 1.5,
        };
        assert!(m.is_match());
        assert_eq!(m.label(), Some("alice"));
        assert_eq!(m.distance(), Some(4.0));

        let below = Recognition::BelowThreshold {
            best: Some(("bob".into(), 42.0)),
        };
        assert!(!below.is_match());
        assert_eq!(below.label(), None);
        assert_eq!(below.distance(), Some(42.0));

        assert_eq!(Recognition::NoCandidates.distance(), None);
    }
}
