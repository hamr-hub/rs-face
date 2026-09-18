//! `ArcFaceRecognizer` — face recognition wired to the pure numerics in
//! [`crate::arcface`].
//!
//! This is the capability the crate previously lacked entirely: turning a detected face
//! into an identity. The full path is
//!
//! ```text
//! RgbImage + Landmarks
//!   └─▶ similarity transform onto the ArcFace canonical layout   (crate::align)
//!         └─▶ 112x112 aligned crop, normalised to [-1, 1]        (crate::arcface)
//!               └─▶ ONNX forward pass                            (this module)
//!                     └─▶ L2-normalised 512-d Embedding          (crate::embedding)
//!                           └─▶ cosine match against a Gallery
//! ```
//!
//! # Landmarks are mandatory, and that is enforced
//!
//! [`ArcFaceRecognizer::embed`] requires [`crate::face::Landmarks`]. A detection without them cannot be aligned, and
//! an unaligned crop produces an embedding that is *syntactically* fine — 512 unit-length
//! floats — but semantically meaningless, yielding cosine similarities barely above chance.
//! Because that failure is invisible at every layer, the type system refuses it: there is
//! no entry point that accepts a bare bounding box.

use crate::arcface;
use crate::embedding::{Embedding, Gallery, MatchOutcome, ARCFACE_EMBEDDING_DIM};
use crate::face::{FaceDetection, Landmarks};
use crate::image::RgbImage;
use crate::models::{ModelKind, ModelSpec};
use crate::onnx::{open_session, Inference, OnnxError, SessionConfig};
use std::path::Path;

/// ArcFace-family recogniser backed by an ONNX session.
pub struct ArcFaceRecognizer {
    session: Box<dyn Inference>,
    /// Embedding dimension reported by the loaded graph.
    dim: usize,
}

impl ArcFaceRecognizer {
    /// Load a recognition model.
    ///
    /// Rejects a spec whose kind is not [`ModelKind::Recognizer`]: loading a *detector*
    /// here would produce nine output tensors that could still be coerced into a vector
    /// and normalised, giving a plausible-looking embedding from entirely the wrong model.
    pub fn open(
        path: &Path,
        spec: Option<&ModelSpec>,
        session_cfg: &SessionConfig,
    ) -> Result<Self, OnnxError> {
        if let Some(s) = spec {
            if s.kind != ModelKind::Recognizer {
                return Err(OnnxError::UnexpectedShape(format!(
                    "model '{}' is a {:?}, not a recogniser",
                    s.id, s.kind
                )));
            }
        }

        let crop = crate::align::ARCFACE_CROP_SIZE;
        let session_cfg = session_cfg.clone().with_input_shape(&[1, 3, crop, crop]);
        let session = open_session(path, spec, &session_cfg)?;

        if session.num_outputs() != 1 {
            return Err(OnnxError::UnexpectedShape(format!(
                "a recognition graph must expose exactly 1 output, found {}",
                session.num_outputs()
            )));
        }

        // Probe once to learn the real embedding dimension, rather than assuming 512.
        // buffalo_s's MobileFaceNet and buffalo_l's R50 both emit 512, but third-party
        // backbones vary, and a mismatched gallery must fail at load, not at match time.
        let probe = vec![0.0f32; 3 * crop * crop];
        let out = session.run(&probe, &[1, 3, crop, crop])?;
        let dim = out
            .first()
            .map(|t| t.data.len())
            .filter(|&d| d > 0)
            .ok_or_else(|| {
                OnnxError::UnexpectedShape("recognition graph produced an empty output".into())
            })?;

        Ok(Self { session, dim })
    }

    /// Convenience: load from a model directory using a registry spec.
    pub fn from_spec(
        dir: &Path,
        spec: &ModelSpec,
        session_cfg: &SessionConfig,
    ) -> Result<Self, OnnxError> {
        Self::open(&dir.join(spec.file_name), Some(spec), session_cfg)
    }

    /// Embedding dimension this model produces.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Whether this model produces the standard 512-d ArcFace embedding.
    pub fn is_standard_arcface_dim(&self) -> bool {
        self.dim == ARCFACE_EMBEDDING_DIM
    }

    /// Embed an already-aligned 112x112 crop.
    ///
    /// Use only when alignment was performed elsewhere; prefer [`Self::embed`].
    pub fn embed_aligned(&self, crop: &RgbImage) -> Result<Embedding, OnnxError> {
        let input = arcface::preprocess_aligned(crop).ok_or_else(|| {
            OnnxError::UnexpectedShape(format!(
                "expected an aligned {size}x{size} crop, got {}x{}",
                crop.width(),
                crop.height(),
                size = crate::align::ARCFACE_CROP_SIZE
            ))
        })?;

        let c = crate::align::ARCFACE_CROP_SIZE;
        let outputs = self.session.run(&input, &[1, 3, c, c])?;
        let raw = &outputs
            .first()
            .ok_or_else(|| OnnxError::Backend("no output tensor".into()))?
            .data;

        arcface::postprocess(raw).ok_or_else(|| {
            // A collapsed output is a real, diagnosable condition (usually a black or
            // NaN crop), so say so rather than returning a zero vector that would match
            // every gallery entry.
            OnnxError::Backend(
                "recognition output collapsed to a zero or non-finite vector; the crop was \
                 probably blank or the alignment degenerate"
                    .into(),
            )
        })
    }

    /// Align a face using its landmarks and embed it. The normal entry point.
    pub fn embed(&self, img: &RgbImage, lms: &Landmarks) -> Result<Embedding, OnnxError> {
        let crop = crate::align::norm_crop(img, lms).ok_or_else(|| {
            OnnxError::UnexpectedShape(
                "landmarks are degenerate (coincident points); cannot align this face".into(),
            )
        })?;
        self.embed_aligned(&crop)
    }

    /// Embed a detection, requiring that it carries landmarks.
    ///
    /// Returns `Ok(None)` when the detection has no landmarks. `None` rather than an error
    /// because a mixed batch from a keypoint-less detector is a legitimate situation that
    /// the caller should skip over, not a failure.
    pub fn embed_detection(
        &self,
        img: &RgbImage,
        det: &FaceDetection,
    ) -> Result<Option<Embedding>, OnnxError> {
        match det.landmarks {
            None => Ok(None),
            Some(lms) => self.embed(img, &lms).map(Some),
        }
    }

    /// Embed every landmark-bearing detection in a frame.
    ///
    /// Returns `(index, embedding)` pairs so callers can tie an embedding back to its
    /// detection; detections without landmarks are skipped rather than silently misaligned.
    pub fn embed_all(
        &self,
        img: &RgbImage,
        dets: &[FaceDetection],
    ) -> Result<Vec<(usize, Embedding)>, OnnxError> {
        let mut out = Vec::new();
        for (i, d) in dets.iter().enumerate() {
            if let Some(e) = self.embed_detection(img, d)? {
                out.push((i, e));
            }
        }
        Ok(out)
    }

    /// Enrol a face into `gallery` under `label`.
    pub fn enroll(
        &self,
        gallery: &mut Gallery,
        label: impl Into<String>,
        img: &RgbImage,
        lms: &Landmarks,
    ) -> Result<(), OnnxError> {
        let e = self.embed(img, lms)?;
        gallery.enroll(label, e);
        Ok(())
    }

    /// Identify a face against `gallery`.
    pub fn identify(
        &self,
        gallery: &Gallery,
        img: &RgbImage,
        lms: &Landmarks,
    ) -> Result<MatchOutcome, OnnxError> {
        Ok(gallery.identify(&self.embed(img, lms)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{License, ModelKind};

    fn detector_spec() -> ModelSpec {
        ModelSpec {
            id: "a_detector",
            kind: ModelKind::Detector,
            url: "https://example.com/x.onnx",
            file_name: "x.onnx",
            sha256: None,
            size_bytes: None,
            license: License::Permissive("MIT"),
            input_size: 640,
            accuracy: "",
        }
    }

    /// Loading a detector as a recogniser must be refused up front. Nine SCRFD output
    /// tensors could otherwise be flattened and L2-normalised into something that looks
    /// exactly like a valid embedding.
    #[test]
    fn refuses_a_detector_spec() {
        let spec = detector_spec();
        let err = ArcFaceRecognizer::open(
            Path::new("/definitely/not/here.onnx"),
            Some(&spec),
            &SessionConfig::default(),
        )
        .err()
        .expect("must refuse a detector");
        match err {
            OnnxError::UnexpectedShape(m) => {
                assert!(m.contains("not a recogniser"), "got {m}");
            }
            other => panic!("expected a kind-mismatch error, got {other:?}"),
        }
    }

    #[test]
    fn opening_a_missing_model_reports_cleanly() {
        let err = ArcFaceRecognizer::open(
            Path::new("/definitely/not/here.onnx"),
            None,
            &SessionConfig::default(),
        )
        .err()
        .expect("must fail");
        assert!(
            matches!(err, OnnxError::ModelNotFound(_) | OnnxError::NoBackend),
            "got {err:?}"
        );
    }

    #[test]
    fn registry_recognizers_are_accepted_by_the_kind_check() {
        // The kind guard must not reject the models we actually ship support for; a
        // missing file is the expected failure, not a kind mismatch.
        for spec in crate::models::REGISTRY
            .iter()
            .filter(|m| m.kind == ModelKind::Recognizer)
        {
            let err = ArcFaceRecognizer::open(
                Path::new("/definitely/not/here.onnx"),
                Some(spec),
                &SessionConfig::default(),
            )
            .err()
            .expect("must fail on a missing file");
            assert!(
                !matches!(&err, OnnxError::UnexpectedShape(m) if m.contains("not a recogniser")),
                "{} was wrongly rejected as a non-recogniser",
                spec.id
            );
        }
    }

    #[test]
    fn standard_dim_predicate_matches_the_gallery_constant() {
        assert_eq!(ARCFACE_EMBEDDING_DIM, 512);
    }
}
