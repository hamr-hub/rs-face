//! `LivenessDetector` — MiniFASNet inference wired to the pure logic in
//! [`crate::liveness`].
//!
//! This is the thin layer that turns the two published MiniFASNet ONNX graphs
//! into a real/spoof verdict. It owns exactly three responsibilities:
//!
//! 1. Hold one session per crop scale and run the forward passes.
//! 2. Turn each graph's three logits into a softmax probability row.
//! 3. Fuse the rows and report honestly what it decided.
//!
//! The crop-expansion, BGR/NCHW build, softmax and fusion decision all live
//! in [`crate::liveness`] as runtime-free functions; this module only supplies
//! the missing forward pass, keeping the boundary sharp in the same spirit as
//! `crate::scrfd_detector` (ONNX-feature gated).
//!
//! # Two models are loaded, not one
//!
//! MiniFASNetV2 inspects a tight 2.7× crop and MiniFASNetV1SE a wider 4.0×
//! crop; their softmax outputs are averaged before the decision, which is the
//! upstream-recommended ensemble and materially steadier than either model
//! alone. Loading a single model is supported via [`Self::open_single`].

use crate::face::Detection;
use crate::image::RgbImage;
use crate::liveness::{
    self, decide, expanded_crop, preprocess_bgr, softmax, LivenessConfig, LivenessOutcome,
    CROP_SCALES, INPUT, LOW_QUALITY_CLASS, NUM_CLASSES,
};
use crate::models::{ModelKind, ModelSpec};
use crate::onnx::{open_session, Inference, OnnxError, SessionConfig};
use crate::quality::{assess as assess_quality, QualityConfig};
use std::path::Path;

/// One loaded MiniFASNet graph paired with its crop scale.
struct Head {
    session: Box<dyn Inference>,
    scale: f32,
}

/// Silent face-anti-spoofing detector backed by one or two ONNX sessions.
pub struct LivenessDetector {
    heads: Vec<Head>,
    config: LivenessConfig,
    quality: QualityConfig,
}

impl LivenessDetector {
    /// Load the canonical two-model detector.
    ///
    /// `path_v2` / `path_v1se` point at the MiniFASNetV2 (2.7×) and
    /// MiniFASNetV1SE (4.0×) ONNX files; the matching specs supply pinned
    /// digests and are checked to be [`ModelKind::Liveness`]. Pass `None`
    /// specs to use your own exports with integrity relaxed accordingly.
    pub fn open(
        path_v2: &Path,
        spec_v2: Option<&ModelSpec>,
        path_v1se: &Path,
        spec_v1se: Option<&ModelSpec>,
        session_cfg: &SessionConfig,
        config: LivenessConfig,
        quality: QualityConfig,
    ) -> Result<Self, OnnxError> {
        let heads = vec![
            open_head(path_v2, spec_v2, session_cfg, CROP_SCALES[0])?,
            open_head(path_v1se, spec_v1se, session_cfg, CROP_SCALES[1])?,
        ];
        Ok(Self {
            heads,
            config,
            quality,
        })
    }

    /// Load a single-model detector (one crop scale).
    pub fn open_single(
        path: &Path,
        spec: Option<&ModelSpec>,
        session_cfg: &SessionConfig,
        scale: f32,
        config: LivenessConfig,
        quality: QualityConfig,
    ) -> Result<Self, OnnxError> {
        let heads = vec![open_head(path, spec, session_cfg, scale)?];
        Ok(Self {
            heads,
            config,
            quality,
        })
    }

    /// Number of models fused for each decision.
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }

    /// Run the liveness check on one detected face.
    ///
    /// Each head expands the detection by its crop scale (clamped to the image
    /// bounds), crops and resizes to 80×80, builds the raw BGR tensor, runs the
    /// graph and softmax-normalises its logits. The rows are then averaged and
    /// rendered through [`LivenessConfig`].
    pub fn check(&self, img: &RgbImage, det: &Detection) -> Result<LivenessOutcome, OnnxError> {
        // Quality gate first: measure on the native-resolution crop (not the
        // 80×80 resize), so blur / tiny size / clipping are still visible. A
        // poor crop is rejected fail-closed without spending a forward pass.
        if self.quality.enabled {
            let (qx, qy, qw, qh) = expanded_crop(img.width(), img.height(), det, CROP_SCALES[0]);
            if qw == 0 || qh == 0 {
                return Err(OnnxError::UnexpectedShape(
                    "quality-gate crop is empty".into(),
                ));
            }
            let native_crop = img.crop(qx, qy, qw, qh);
            let gray = native_crop.to_gray();
            let report = assess_quality(&gray, &self.quality);
            if !report.acceptable {
                return Ok(LivenessOutcome {
                    is_real: false,
                    real_score: 0.0,
                    probs: [0.0; NUM_CLASSES],
                    class: LOW_QUALITY_CLASS,
                });
            }
        }

        let mut rows = Vec::with_capacity(self.heads.len());
        for head in &self.heads {
            let (cx, cy, cw, ch) = expanded_crop(img.width(), img.height(), det, head.scale);
            if cw == 0 || ch == 0 {
                return Err(OnnxError::UnexpectedShape(
                    "expanded liveness crop is empty".into(),
                ));
            }
            let crop = img.crop(cx, cy, cw, ch);
            let resized = if crop.width() == INPUT && crop.height() == INPUT {
                crop
            } else {
                crop.resize_bilinear(INPUT, INPUT)
            };
            let input = preprocess_bgr(&resized).ok_or_else(|| {
                OnnxError::UnexpectedShape("liveness crop did not yield an 80x80 tensor".into())
            })?;

            let shape = [1usize, 3, INPUT, INPUT];
            let outputs = head.session.run(&input, &shape)?;
            let tensor = outputs.first().ok_or_else(|| {
                OnnxError::UnexpectedShape("liveness graph produced no output".into())
            })?;
            if tensor.data.len() < NUM_CLASSES {
                return Err(OnnxError::UnexpectedShape(format!(
                    "liveness graph output has {} values, expected {NUM_CLASSES}",
                    tensor.data.len()
                )));
            }
            let probs = softmax(&tensor.data[..NUM_CLASSES])
                .ok_or_else(|| OnnxError::UnexpectedShape("non-finite liveness logits".into()))?;
            rows.push(probs);
        }
        decide(&rows, &self.config)
            .ok_or_else(|| OnnxError::UnexpectedShape("could not fuse liveness rows".into()))
    }
}

/// Open and validate one MiniFASNet session for a given crop scale.
fn open_head(
    path: &Path,
    spec: Option<&ModelSpec>,
    session_cfg: &SessionConfig,
    scale: f32,
) -> Result<Head, OnnxError> {
    if let Some(s) = spec {
        if s.kind != ModelKind::Liveness {
            return Err(OnnxError::UnexpectedShape(format!(
                "model '{}' is a {:?}, not a liveness model",
                s.id, s.kind
            )));
        }
    }
    let session_cfg = session_cfg.clone().with_input_shape(&[1, 3, INPUT, INPUT]);
    let session = open_session(path, spec, &session_cfg)?;

    if session.num_outputs() != 1 {
        return Err(OnnxError::UnexpectedShape(format!(
            "a liveness graph must expose exactly 1 output, found {}",
            session.num_outputs()
        )));
    }
    // Probe once to confirm the output is a 3-wide head, so a wrong export
    // fails at load rather than producing a verdict from the wrong values.
    let probe = vec![0.0f32; 3 * INPUT * INPUT];
    let out = session.run(&probe, &[1, 3, INPUT, INPUT])?;
    let width = out.first().map(|t| t.data.len()).unwrap_or(0);
    if width < NUM_CLASSES {
        return Err(OnnxError::UnexpectedShape(format!(
            "liveness output has {width} values, expected {NUM_CLASSES}"
        )));
    }
    Ok(Head { session, scale })
}
