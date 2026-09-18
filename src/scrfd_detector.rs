//! `ScrfdDetector` — SCRFD inference wired to the pure decoding in [`crate::scrfd`].
//!
//! This is the thin layer that turns "a graph plus some pure functions" into a
//! [`FaceDetector`]. It owns three responsibilities and nothing else:
//!
//! 1. Hold a session and run the forward pass.
//! 2. Work out how the graph's output tensors map onto SCRFD's per-stride heads.
//! 3. Report honestly what it is — [`Maturity::Production`], RGB-native, and whether the
//!    loaded graph actually has a keypoint head.
//!
//! # Output ordering is discovered, not assumed
//!
//! ONNX does not guarantee output order, and SCRFD exports vary: some emit
//! `[score8, score16, score32, bbox8, bbox16, bbox32, kps8, kps16, kps32]` (grouped by
//! head), others `[score8, bbox8, kps8, score16, ...]` (grouped by stride). Guessing wrong
//! does not error — it pairs scores with box tensors from a different stride and produces
//! boxes scattered across the frame. So [`HeadLayout::infer`] derives the grouping from the
//! tensors' trailing dimensions (1 for scores, 4 for boxes, 10 for keypoints) and their
//! lengths, which is self-describing and testable without a model file.

use crate::detector::Detection;
use crate::face::FaceDetection;
use crate::face_detector::{ColorInput, FaceDetector, Maturity};
use crate::image::{GrayImage, RgbImage};
use crate::models::ModelSpec;
use crate::onnx::{open_session, Inference, OnnxError, SessionConfig, Tensor};
use crate::scrfd::{self, ScrfdConfig, StrideOutput, NUM_ANCHORS, STRIDES};
use std::path::Path;

/// Trailing-dimension widths that identify each SCRFD head.
const SCORE_DIM: usize = 1;
const BBOX_DIM: usize = 4;
const KPS_DIM: usize = 10;

/// How a graph's flat output list maps onto (stride, head) pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadLayout {
    /// All scores, then all boxes, then all keypoints. The common InsightFace export.
    GroupedByHead,
    /// Per stride: score, bbox, keypoints. Seen in some re-exports.
    GroupedByStride,
}

impl HeadLayout {
    /// Infer the layout from the output tensors' trailing dimensions.
    ///
    /// Returns `None` when the pattern matches neither layout, which is strictly better
    /// than defaulting: an unrecognised export must be refused loudly rather than decoded
    /// into plausible-looking nonsense.
    pub fn infer(outputs: &[Tensor]) -> Option<Self> {
        let n = STRIDES.len();
        let dims: Vec<usize> = outputs.iter().map(|t| t.last_dim()).collect();

        // Grouped by head: the first n tensors are all scores.
        if dims.len() >= n * 2 && dims[..n].iter().all(|&d| d == SCORE_DIM) {
            let boxes_ok = dims[n..n * 2].iter().all(|&d| d == BBOX_DIM);
            if boxes_ok {
                return Some(HeadLayout::GroupedByHead);
            }
        }

        // Grouped by stride: repeating (1, 4[, 10]).
        let per = if dims.len() == n * 3 { 3 } else { 2 };
        if dims.len() == n * per {
            let ok = (0..n).all(|s| {
                dims[s * per] == SCORE_DIM
                    && dims[s * per + 1] == BBOX_DIM
                    && (per == 2 || dims[s * per + 2] == KPS_DIM)
            });
            if ok {
                return Some(HeadLayout::GroupedByStride);
            }
        }
        None
    }

    /// Index of the (stride, head) tensor in the flat output list.
    ///
    /// `head` is 0 = score, 1 = bbox, 2 = kps.
    pub fn index(&self, stride_idx: usize, head: usize, has_kps: bool) -> usize {
        let n = STRIDES.len();
        match self {
            HeadLayout::GroupedByHead => head * n + stride_idx,
            HeadLayout::GroupedByStride => {
                let per = if has_kps { 3 } else { 2 };
                stride_idx * per + head
            }
        }
    }
}

/// SCRFD face detector backed by an ONNX session.
pub struct ScrfdDetector {
    session: Box<dyn Inference>,
    config: ScrfdConfig,
    layout: HeadLayout,
    has_kps: bool,
}

impl ScrfdDetector {
    /// Load a SCRFD model from `path`.
    ///
    /// `spec` supplies the pinned digest and input size; pass `None` to use your own
    /// export with integrity checking relaxed accordingly.
    pub fn open(
        path: &Path,
        spec: Option<&ModelSpec>,
        session_cfg: &SessionConfig,
        mut config: ScrfdConfig,
    ) -> Result<Self, OnnxError> {
        if let Some(s) = spec {
            config.input_size = s.input_size;
        }
        // Tell the backend our concrete input shape: the pure-Rust tract backend cannot
        // resolve a dynamic-axis graph without it, and we are the component that knows.
        let session_cfg =
            session_cfg
                .clone()
                .with_input_shape(&[1, 3, config.input_size, config.input_size]);
        let session = open_session(path, spec, &session_cfg)?;

        let (_, has_kps) = scrfd::probe_outputs(session.num_outputs()).ok_or_else(|| {
            OnnxError::UnexpectedShape(format!(
                "graph exposes {} outputs; SCRFD needs {} (with keypoints) or {} (without)",
                session.num_outputs(),
                STRIDES.len() * 3,
                STRIDES.len() * 2
            ))
        })?;

        // The layout needs real tensors to infer, so probe with a zero input. One wasted
        // forward pass at construction buys certainty for every subsequent frame.
        let probe_shape = [1usize, 3, config.input_size, config.input_size];
        let probe_input = vec![0.0f32; 3 * config.input_size * config.input_size];
        let outputs = session.run(&probe_input, &probe_shape)?;

        for (i, t) in outputs.iter().enumerate() {
            if !t.is_consistent() {
                return Err(OnnxError::UnexpectedShape(format!(
                    "output {i} has shape {:?} but {} elements",
                    t.shape,
                    t.data.len()
                )));
            }
        }

        let layout = HeadLayout::infer(&outputs).ok_or_else(|| {
            OnnxError::UnexpectedShape(format!(
                "could not identify SCRFD head layout from output trailing dims {:?}; \
                 expected scores(1), boxes(4) and optionally keypoints(10) per stride",
                outputs.iter().map(|t| t.last_dim()).collect::<Vec<_>>()
            ))
        })?;

        Ok(Self {
            session,
            config,
            layout,
            has_kps,
        })
    }

    /// Convenience: load using a registry spec and the default model directory.
    pub fn from_spec(
        dir: &Path,
        spec: &ModelSpec,
        session_cfg: &SessionConfig,
    ) -> Result<Self, OnnxError> {
        let path = dir.join(spec.file_name);
        Self::open(&path, Some(spec), session_cfg, ScrfdConfig::default())
    }

    pub fn config(&self) -> &ScrfdConfig {
        &self.config
    }

    pub fn set_config(&mut self, cfg: ScrfdConfig) {
        self.config = cfg;
    }

    /// Whether the loaded graph has a keypoint head, and can therefore feed recognition.
    pub fn has_keypoints(&self) -> bool {
        self.has_kps
    }

    pub fn head_layout(&self) -> HeadLayout {
        self.layout
    }

    /// Detect faces in an RGB frame, returning sub-pixel boxes and landmarks.
    ///
    /// Errors are returned rather than swallowed into an empty `Vec`: an inference failure
    /// and "no faces in this frame" are entirely different conditions, and conflating them
    /// is exactly the bug that made the scaffold detectors so misleading.
    pub fn detect_rgb(&self, img: &RgbImage) -> Result<Vec<FaceDetection>, OnnxError> {
        if img.width() == 0 || img.height() == 0 {
            return Ok(Vec::new());
        }

        let size = self.config.input_size;
        let (input, lb) = scrfd::preprocess(img, size);
        let outputs = self.session.run(&input, &[1, 3, size, size])?;

        let expected = STRIDES.len() * if self.has_kps { 3 } else { 2 };
        if outputs.len() != expected {
            return Err(OnnxError::UnexpectedShape(format!(
                "expected {expected} output tensors, got {}",
                outputs.len()
            )));
        }

        // Borrow each stride's tensors according to the discovered layout.
        let mut groups: Vec<StrideOutput<'_>> = Vec::with_capacity(STRIDES.len());
        for s in 0..STRIDES.len() {
            let scores = &outputs[self.layout.index(s, 0, self.has_kps)];
            let bboxes = &outputs[self.layout.index(s, 1, self.has_kps)];
            let kps = if self.has_kps {
                Some(
                    outputs[self.layout.index(s, 2, self.has_kps)]
                        .data
                        .as_slice(),
                )
            } else {
                None
            };
            groups.push(StrideOutput {
                scores: &scores.data,
                bboxes: &bboxes.data,
                kps,
            });
        }

        Ok(scrfd::postprocess(
            &groups,
            &lb,
            img.width(),
            img.height(),
            &self.config,
        ))
    }
}

impl FaceDetector for ScrfdDetector {
    /// Grayscale entry point, required by the trait.
    ///
    /// SCRFD is RGB-native, so this replicates the luminance plane across all three
    /// channels and runs the model off its training distribution — a real accuracy cost.
    /// [`Self::detect_rgb`] should be preferred whenever colour is available; the
    /// [`ColorInput::Rgb`] declaration exists so callers can route around this.
    fn detect(&self, img: &GrayImage) -> Vec<Detection> {
        let mut rgb = RgbImage::new(img.width(), img.height());
        let src = img.as_slice();
        let dst = rgb.as_mut_slice();
        for (i, &v) in src.iter().enumerate() {
            dst[i * 3] = v;
            dst[i * 3 + 1] = v;
            dst[i * 3 + 2] = v;
        }
        // The trait cannot express failure, so an inference error degrades to "no
        // detections" here. Callers that need to distinguish the two must use detect_rgb.
        self.detect_rgb(&rgb)
            .unwrap_or_default()
            .into_iter()
            .map(|d| d.to_detection())
            .collect()
    }

    fn name(&self) -> &'static str {
        "scrfd"
    }

    fn description(&self) -> &'static str {
        "SCRFD-10G (InsightFace): anchor-free, strides 8/16/32, 5-pt keypoints. \
         WIDER FACE AP 0.954/0.940/0.828."
    }

    fn color_input(&self) -> ColorInput {
        ColorInput::Rgb
    }

    fn maturity(&self) -> Maturity {
        Maturity::Production
    }

    fn has_landmarks(&self) -> bool {
        self.has_kps
    }

    fn detect_faces_rgb(&self, img: &RgbImage) -> Vec<FaceDetection> {
        self.detect_rgb(img).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tensor with the given trailing dim and prediction count.
    fn head(preds: usize, dim: usize) -> Tensor {
        Tensor::new(vec![0.0; preds * dim], vec![1, preds, dim])
    }

    /// Output tensors for a 640 input, laid out grouped by head.
    fn grouped_by_head(with_kps: bool) -> Vec<Tensor> {
        let counts: Vec<usize> = STRIDES
            .iter()
            .map(|s| (640 / s) * (640 / s) * NUM_ANCHORS)
            .collect();
        let mut v: Vec<Tensor> = counts.iter().map(|&c| head(c, SCORE_DIM)).collect();
        v.extend(counts.iter().map(|&c| head(c, BBOX_DIM)));
        if with_kps {
            v.extend(counts.iter().map(|&c| head(c, KPS_DIM)));
        }
        v
    }

    fn grouped_by_stride(with_kps: bool) -> Vec<Tensor> {
        let mut v = Vec::new();
        for s in STRIDES {
            let c = (640 / s) * (640 / s) * NUM_ANCHORS;
            v.push(head(c, SCORE_DIM));
            v.push(head(c, BBOX_DIM));
            if with_kps {
                v.push(head(c, KPS_DIM));
            }
        }
        v
    }

    #[test]
    fn infers_grouped_by_head_layout() {
        assert_eq!(
            HeadLayout::infer(&grouped_by_head(true)),
            Some(HeadLayout::GroupedByHead)
        );
        assert_eq!(
            HeadLayout::infer(&grouped_by_head(false)),
            Some(HeadLayout::GroupedByHead)
        );
    }

    #[test]
    fn infers_grouped_by_stride_layout() {
        assert_eq!(
            HeadLayout::infer(&grouped_by_stride(true)),
            Some(HeadLayout::GroupedByStride)
        );
        assert_eq!(
            HeadLayout::infer(&grouped_by_stride(false)),
            Some(HeadLayout::GroupedByStride)
        );
    }

    #[test]
    fn refuses_an_unrecognised_layout_rather_than_guessing() {
        // Trailing dims that match no SCRFD head pattern.
        let weird = vec![head(100, 7), head(100, 7), head(100, 3)];
        assert!(
            HeadLayout::infer(&weird).is_none(),
            "an unknown export must be refused, not decoded into nonsense"
        );
        assert!(HeadLayout::infer(&[]).is_none());
    }

    #[test]
    fn grouped_by_head_indices_map_correctly() {
        let l = HeadLayout::GroupedByHead;
        // scores: 0,1,2  boxes: 3,4,5  kps: 6,7,8
        assert_eq!(
            (
                l.index(0, 0, true),
                l.index(1, 0, true),
                l.index(2, 0, true)
            ),
            (0, 1, 2)
        );
        assert_eq!((l.index(0, 1, true), l.index(2, 1, true)), (3, 5));
        assert_eq!((l.index(0, 2, true), l.index(2, 2, true)), (6, 8));
    }

    #[test]
    fn grouped_by_stride_indices_map_correctly() {
        let l = HeadLayout::GroupedByStride;
        // stride0: 0,1,2  stride1: 3,4,5  stride2: 6,7,8
        assert_eq!(
            (
                l.index(0, 0, true),
                l.index(0, 1, true),
                l.index(0, 2, true)
            ),
            (0, 1, 2)
        );
        assert_eq!(l.index(1, 0, true), 3);
        assert_eq!(l.index(2, 2, true), 8);
        // Without keypoints the stride span narrows to 2.
        assert_eq!(l.index(1, 0, false), 2);
        assert_eq!(l.index(2, 1, false), 5);
    }

    /// The two layouts must genuinely disagree, otherwise inferring one is pointless and
    /// a wrong inference would be harmless — which would mean this whole mechanism is
    /// untested dead weight.
    #[test]
    fn the_two_layouts_produce_different_indices() {
        let a = HeadLayout::GroupedByHead;
        let b = HeadLayout::GroupedByStride;
        let differs =
            (0..STRIDES.len()).any(|s| (0..3).any(|h| a.index(s, h, true) != b.index(s, h, true)));
        assert!(differs, "layouts must not be interchangeable");
    }

    #[test]
    fn index_covers_every_slot_exactly_once() {
        // A permutation bug would show up as a duplicated or missing index.
        for layout in [HeadLayout::GroupedByHead, HeadLayout::GroupedByStride] {
            for has_kps in [true, false] {
                let heads = if has_kps { 3 } else { 2 };
                let mut seen: Vec<usize> = (0..STRIDES.len())
                    .flat_map(|s| (0..heads).map(move |h| (s, h)))
                    .map(|(s, h)| layout.index(s, h, has_kps))
                    .collect();
                let n = seen.len();
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(
                    seen.len(),
                    n,
                    "{layout:?} has_kps={has_kps} duplicated an index"
                );
                assert_eq!(seen, (0..n).collect::<Vec<_>>(), "{layout:?} index gap");
            }
        }
    }

    #[test]
    fn opening_a_missing_model_reports_cleanly() {
        let r = ScrfdDetector::open(
            Path::new("/definitely/not/here.onnx"),
            None,
            &SessionConfig::default(),
            ScrfdConfig::default(),
        );
        let err = r.err().expect("must fail");
        // Zero-dep builds report NoBackend; ONNX builds report ModelNotFound.
        assert!(
            matches!(err, OnnxError::ModelNotFound(_) | OnnxError::NoBackend),
            "got {err:?}"
        );
    }
}
