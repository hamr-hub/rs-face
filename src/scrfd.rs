//! SCRFD: pre-processing, anchor-free decoding, and post-processing.
//!
//! SCRFD (Sample and Computation Redistribution for Face Detection, Guo et al. 2021) is
//! the accuracy leader among openly published face detectors: the 10G variant reaches
//! **0.954 / 0.940 / 0.828** AP on WIDER FACE easy/medium/hard, against roughly 0.30 hard
//! AP for the Viola-Jones cascade this crate started with.
//!
//! Everything in this module is **pure**: it turns pixels into an input tensor and turns
//! raw output tensors into [`FaceDetection`]s, with no reference to any inference runtime.
//! That separation is deliberate — the decoding is where the subtle, accuracy-destroying
//! bugs live, so it is unit-testable against synthetic tensors without a 17 MB model file
//! or an ONNX Runtime install. The `ort`/`tract` backends only supply the forward pass.
//!
//! # Geometry, stated precisely
//!
//! SCRFD is **anchor-free**. For each FPN stride `s` in `{8, 16, 32}` the network emits a
//! `(H/s) x (W/s)` grid, with `num_anchors` predictions per cell. Each prediction is four
//! *distances* from the cell's anchor centre to the box edges, expressed in units of `s`.
//!
//! The one detail worth stating explicitly, because published descriptions disagree:
//! the anchor centre is `(col * s, row * s)` — **not** `(col * s + s/2, row * s + s/2)`.
//! InsightFace's `scrfd.py` builds centres as `mgrid * stride` with no half-stride offset.
//! Adding `s/2` looks like a reasonable "pixel centre" correction and produces boxes that
//! still land on faces, but every one is displaced by 4, 8 or 16 px depending on stride,
//! which quietly costs both AP and landmark quality. See
//! [`anchor_centers`] and the test `anchor_centers_have_no_half_stride_offset`.

use std::cmp::Ordering;

use crate::face::{FaceDetection, Landmarks, NUM_LANDMARKS};
use crate::image::RgbImage;

/// FPN strides SCRFD emits, ascending. Output tensors arrive grouped in this order.
pub const STRIDES: [usize; 3] = [8, 16, 32];

/// Predictions per grid cell. All published SCRFD variants use 2.
pub const NUM_ANCHORS: usize = 2;

/// Mean subtracted from each channel before scaling.
pub const INPUT_MEAN: f32 = 127.5;

/// Divisor applied after mean subtraction.
///
/// Note this is `128.0`, not `127.5`. SCRFD's published pre-processing is
/// `blobFromImage(scalefactor=1/128, mean=(127.5,127.5,127.5))`, so the input range is
/// `[-0.996, 0.996)` rather than exactly `[-1, 1]`. The recognition backbone uses a
/// *different* divisor — see [`crate::arcface::INPUT_STD`]. Sharing one constant between
/// them would be wrong.
pub const INPUT_STD: f32 = 128.0;

/// Configuration for SCRFD post-processing.
#[derive(Clone, Debug)]
pub struct ScrfdConfig {
    /// Minimum objectness score to keep a box.
    ///
    /// 0.5 is the InsightFace default for detection. Lower it toward 0.3 to trade
    /// precision for recall on crowded or low-resolution scenes.
    pub score_threshold: f32,
    /// IoU threshold for NMS. 0.4 matches InsightFace.
    pub nms_iou: f32,
    /// Square network input resolution. Must match the exported model.
    pub input_size: usize,
    /// Drop boxes whose shorter side is below this many pixels, measured in the
    /// *original* image, after scaling back from the letterbox.
    pub min_face_size: f32,
}

impl Default for ScrfdConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            nms_iou: 0.4,
            input_size: 640,
            min_face_size: 0.0,
        }
    }
}

impl ScrfdConfig {
    pub fn with_score_threshold(mut self, t: f32) -> Self {
        self.score_threshold = t;
        self
    }
    pub fn with_nms_iou(mut self, t: f32) -> Self {
        self.nms_iou = t;
        self
    }
    pub fn with_input_size(mut self, s: usize) -> Self {
        self.input_size = s;
        self
    }
    pub fn with_min_face_size(mut self, s: f32) -> Self {
        self.min_face_size = s;
        self
    }
}

/// The letterbox mapping from an original frame into the square network input.
///
/// SCRFD's reference implementation resizes preserving aspect ratio and pads at the
/// **bottom and right only**, leaving the image origin at `(0, 0)`. That is why this
/// struct carries a scale but no offset: recovering original coordinates is a single
/// division, with no translation term to get backwards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    /// Uniform factor applied to the original image to fit the input square.
    pub scale: f32,
    /// Width of the live (non-padded) region, in input-tensor pixels.
    pub resized_w: usize,
    /// Height of the live region, in input-tensor pixels.
    pub resized_h: usize,
    /// Square network input size.
    pub input_size: usize,
}

impl Letterbox {
    /// Compute the mapping for an image of `w` x `h` into an `input_size` square.
    pub fn compute(w: usize, h: usize, input_size: usize) -> Self {
        if w == 0 || h == 0 {
            return Self {
                scale: 1.0,
                resized_w: 0,
                resized_h: 0,
                input_size,
            };
        }
        // Fit the longer side; the shorter side is padded.
        let scale = (input_size as f32 / w as f32).min(input_size as f32 / h as f32);
        Self {
            scale,
            // Truncate, matching cv2.resize on an int target computed the same way.
            resized_w: (w as f32 * scale) as usize,
            resized_h: (h as f32 * scale) as usize,
            input_size,
        }
    }

    /// Map an x or y coordinate from input-tensor space back to the original image.
    ///
    /// Named `unscale` rather than `to_original` on purpose: [`Letterbox`] is `Copy`, so
    /// clippy's `wrong_self_convention` reads a `to_*` method as converting `self`.
    #[inline]
    pub fn unscale(&self, v: f32) -> f32 {
        v / self.scale
    }
}

/// Build the SCRFD input tensor: NCHW `f32`, RGB, `(x - 127.5) / 128`.
///
/// Returns the flat tensor of length `3 * input_size * input_size` alongside the
/// [`Letterbox`] needed to map detections back. Padded regions are filled with the
/// normalised value of pixel 0 (i.e. `-127.5/128`), matching a zero-padded uint8 buffer
/// as the reference implementation produces — padding with normalised *zero* instead
/// would inject a grey border the model never saw during training.
pub fn preprocess(img: &RgbImage, input_size: usize) -> (Vec<f32>, Letterbox) {
    let lb = Letterbox::compute(img.width(), img.height(), input_size);
    let plane = input_size * input_size;
    // Padding value: raw byte 0 pushed through the same normalisation.
    let pad = (0.0 - INPUT_MEAN) / INPUT_STD;
    let mut out = vec![pad; 3 * plane];

    if lb.resized_w == 0 || lb.resized_h == 0 {
        return (out, lb);
    }

    let (sw, sh) = (img.width(), img.height());
    let src = img.as_slice();

    for y in 0..lb.resized_h.min(input_size) {
        // Nearest-neighbour sampling of the source row/column. SCRFD is robust to the
        // interpolation kernel at this stage; bilinear costs ~3x for no measurable AP.
        let sy = ((y as f32 + 0.5) / lb.scale) as usize;
        let sy = sy.min(sh - 1);
        for x in 0..lb.resized_w.min(input_size) {
            let sx = ((x as f32 + 0.5) / lb.scale) as usize;
            let sx = sx.min(sw - 1);
            let si = (sy * sw + sx) * 3;
            let di = y * input_size + x;
            // NCHW: channel-major planes, so R/G/B land a full plane apart.
            for c in 0..3 {
                out[c * plane + di] = (src[si + c] as f32 - INPUT_MEAN) / INPUT_STD;
            }
        }
    }
    (out, lb)
}

/// Generate anchor centres for one stride, in input-tensor pixel coordinates.
///
/// Ordering is row-major over the grid, with the `NUM_ANCHORS` predictions for a cell
/// **adjacent** to each other: `[(r0,c0)a0, (r0,c0)a1, (r0,c1)a0, (r0,c1)a1, ...]`. This
/// interleaving mirrors `np.stack([centers] * num_anchors, axis=1).reshape(-1, 2)` in the
/// reference implementation. Grouping all `a0` first instead would pair every score with
/// the wrong spatial cell — producing boxes scattered across the image rather than an
/// obvious error.
///
/// Note there is **no half-stride offset**; see the module docs.
pub fn anchor_centers(grid_w: usize, grid_h: usize, stride: usize) -> Vec<(f32, f32)> {
    let mut v = Vec::with_capacity(grid_w * grid_h * NUM_ANCHORS);
    for row in 0..grid_h {
        for col in 0..grid_w {
            let c = ((col * stride) as f32, (row * stride) as f32);
            for _ in 0..NUM_ANCHORS {
                v.push(c);
            }
        }
    }
    v
}

/// Decode one stride's raw outputs into detections in **input-tensor** coordinates.
///
/// `scores` has one entry per prediction; `bboxes` has 4 (`l, t, r, b` distances in units
/// of `stride`); `kps`, when present, has 10 (five `(dx, dy)` offsets, also in stride
/// units). Returns only predictions above `score_threshold`.
///
/// Distances are converted as `x1 = cx - l*s`, `x2 = cx + r*s` — left/top *subtract*,
/// right/bottom *add*. Landmark offsets always *add*.
pub fn decode_stride(
    scores: &[f32],
    bboxes: &[f32],
    kps: Option<&[f32]>,
    grid_w: usize,
    grid_h: usize,
    stride: usize,
    score_threshold: f32,
) -> Vec<FaceDetection> {
    let centers = anchor_centers(grid_w, grid_h, stride);
    // Trust the shortest tensor rather than the grid: a model whose real output length
    // disagrees with our computed grid must not cause an out-of-bounds panic.
    let n = centers.len().min(scores.len()).min(bboxes.len() / 4);
    let s = stride as f32;

    let mut out = Vec::new();
    for i in 0..n {
        let score = scores[i];
        // `partial_cmp` (not `<`) so a NaN score is rejected: with NaN, `score < threshold`
        // is false, which would let garbage through the gate.
        if !matches!(
            score.partial_cmp(&score_threshold),
            Some(Ordering::Greater | Ordering::Equal)
        ) {
            continue;
        }
        let (cx, cy) = centers[i];
        let b = &bboxes[i * 4..i * 4 + 4];

        let landmarks = kps.and_then(|k| {
            let base = i * NUM_LANDMARKS * 2;
            if base + NUM_LANDMARKS * 2 > k.len() {
                return None;
            }
            let mut pts = [(0.0f32, 0.0f32); NUM_LANDMARKS];
            for (j, p) in pts.iter_mut().enumerate() {
                *p = (cx + k[base + j * 2] * s, cy + k[base + j * 2 + 1] * s);
            }
            Some(Landmarks { points: pts })
        });

        out.push(FaceDetection {
            x1: cx - b[0] * s,
            y1: cy - b[1] * s,
            x2: cx + b[2] * s,
            y2: cy + b[3] * s,
            score,
            landmarks,
        });
    }
    out
}

/// Map a detection from input-tensor coordinates back to the original image.
pub fn to_original(det: FaceDetection, lb: &Letterbox) -> FaceDetection {
    FaceDetection {
        x1: lb.unscale(det.x1),
        y1: lb.unscale(det.y1),
        x2: lb.unscale(det.x2),
        y2: lb.unscale(det.y2),
        score: det.score,
        // Landmarks scale by the same factor. `scaled` is about the origin, which is
        // correct precisely because the letterbox pads bottom-right and keeps origin (0,0).
        landmarks: det.landmarks.map(|l| l.scaled(1.0 / lb.scale)),
    }
}

/// Raw output tensors for a single stride.
pub struct StrideOutput<'a> {
    pub scores: &'a [f32],
    pub bboxes: &'a [f32],
    pub kps: Option<&'a [f32]>,
}

/// Full post-processing: decode all strides, map back, filter, and NMS.
///
/// `orig_w` / `orig_h` are the original frame dimensions, used to clamp boxes at the end
/// (after NMS, so edge faces are not distorted while their IoU is being computed).
pub fn postprocess(
    outputs: &[StrideOutput<'_>],
    lb: &Letterbox,
    orig_w: usize,
    orig_h: usize,
    cfg: &ScrfdConfig,
) -> Vec<FaceDetection> {
    let mut all = Vec::new();
    for (idx, out) in outputs.iter().enumerate() {
        let Some(&stride) = STRIDES.get(idx) else {
            break;
        };
        // Recover the grid from the input size; ONNX exports do not name it.
        let grid = cfg.input_size / stride;
        let mut dets = decode_stride(
            out.scores,
            out.bboxes,
            out.kps,
            grid,
            grid,
            stride,
            cfg.score_threshold,
        );
        for d in dets.drain(..) {
            all.push(to_original(d, lb));
        }
    }

    if cfg.min_face_size > 0.0 {
        all.retain(|d| d.width().min(d.height()) >= cfg.min_face_size);
    }

    let kept = crate::face::nms(all, cfg.nms_iou);
    kept.into_iter()
        .map(|d| d.clamped(orig_w as f32, orig_h as f32))
        .collect()
}

/// Given the number of output tensors a loaded model exposes, report whether it has a
/// keypoint head and how many strides it uses.
///
/// SCRFD exports as 3 tensors per stride with keypoints (score/bbox/kps) or 2 without.
/// Determining this from the *loaded graph* rather than the filename matters: the
/// `det_10g.onnx` inside InsightFace's `buffalo_l` pack and the standalone
/// `scrfd_10g_gnkps.onnx` differ in exactly this respect, and third-party mirrors rename
/// files freely. A wrong guess here silently mis-slices every tensor.
pub fn probe_outputs(num_outputs: usize) -> Option<(usize, bool)> {
    let n_strides = STRIDES.len();
    if num_outputs == n_strides * 3 {
        Some((n_strides, true))
    } else if num_outputs == n_strides * 2 {
        Some((n_strides, false))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Letterbox ----------------------------------------------------------

    #[test]
    fn letterbox_square_image_fills_input_exactly() {
        let lb = Letterbox::compute(640, 640, 640);
        assert_eq!(lb.scale, 1.0);
        assert_eq!((lb.resized_w, lb.resized_h), (640, 640));
    }

    #[test]
    fn letterbox_landscape_pads_vertically() {
        // 1280x720 into 640: scale 0.5 -> 640x360, pad 280 rows at the bottom.
        let lb = Letterbox::compute(1280, 720, 640);
        assert!((lb.scale - 0.5).abs() < 1e-6);
        assert_eq!((lb.resized_w, lb.resized_h), (640, 360));
    }

    #[test]
    fn letterbox_portrait_pads_horizontally() {
        let lb = Letterbox::compute(720, 1280, 640);
        assert!((lb.scale - 0.5).abs() < 1e-6);
        assert_eq!((lb.resized_w, lb.resized_h), (360, 640));
    }

    #[test]
    fn letterbox_preserves_aspect_ratio() {
        let (w, h) = (1920usize, 1080usize);
        let lb = Letterbox::compute(w, h, 640);
        let orig = w as f32 / h as f32;
        let got = lb.resized_w as f32 / lb.resized_h as f32;
        assert!((orig - got).abs() < 0.01, "aspect changed: {orig} vs {got}");
    }

    #[test]
    fn letterbox_upscales_small_images() {
        let lb = Letterbox::compute(160, 160, 640);
        assert!((lb.scale - 4.0).abs() < 1e-6);
    }

    #[test]
    fn letterbox_roundtrip_recovers_coordinates() {
        let lb = Letterbox::compute(1280, 720, 640);
        // A point at x=100 in the original maps to 50 in tensor space and back.
        assert!((lb.unscale(100.0 * lb.scale) - 100.0).abs() < 1e-4);
    }

    #[test]
    fn letterbox_zero_size_does_not_divide_by_zero() {
        let lb = Letterbox::compute(0, 0, 640);
        assert_eq!(lb.resized_w, 0);
        assert!(lb.scale.is_finite());
    }

    // -- Anchor centres -----------------------------------------------------

    /// The load-bearing geometric fact of the whole module.
    #[test]
    fn anchor_centers_have_no_half_stride_offset() {
        let c = anchor_centers(2, 2, 8);
        // First cell must sit exactly on the origin, not at (4, 4).
        assert_eq!(c[0], (0.0, 0.0));
        assert_ne!(c[0], (4.0, 4.0), "half-stride offset must not be applied");
    }

    #[test]
    fn anchor_centers_interleave_anchors_within_a_cell() {
        let c = anchor_centers(3, 2, 8);
        assert_eq!(c.len(), 3 * 2 * NUM_ANCHORS);
        // Cell (0,0) twice, then cell (0,1) twice — not all of anchor 0 first.
        assert_eq!(c[0], (0.0, 0.0));
        assert_eq!(c[1], (0.0, 0.0));
        assert_eq!(c[2], (8.0, 0.0));
        assert_eq!(c[3], (8.0, 0.0));
    }

    #[test]
    fn anchor_centers_are_row_major_x_fastest() {
        let c = anchor_centers(3, 2, 16);
        // Row 0 spans x, then row 1 begins.
        assert_eq!(c[0], (0.0, 0.0));
        assert_eq!(c[2], (16.0, 0.0));
        assert_eq!(c[4], (32.0, 0.0));
        assert_eq!(c[6], (0.0, 16.0));
    }

    #[test]
    fn anchor_centers_grid_covers_input_for_every_stride() {
        for stride in STRIDES {
            let grid = 640 / stride;
            let c = anchor_centers(grid, grid, stride);
            assert_eq!(c.len(), grid * grid * NUM_ANCHORS);
            // Last centre is one stride short of the input edge.
            assert_eq!(
                *c.last().unwrap(),
                ((640 - stride) as f32, (640 - stride) as f32)
            );
        }
    }

    // -- Decoding -----------------------------------------------------------

    /// Build single-prediction tensors for a 1x1 grid (2 anchors).
    fn one_cell(
        score: f32,
        b: [f32; 4],
        k: Option<[f32; 10]>,
    ) -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>) {
        // Anchor 0 carries the signal; anchor 1 is below threshold.
        let scores = vec![score, 0.0];
        let mut bboxes = b.to_vec();
        bboxes.extend_from_slice(&[0.0; 4]);
        let kps = k.map(|k| {
            let mut v = k.to_vec();
            v.extend_from_slice(&[0.0; 10]);
            v
        });
        (scores, bboxes, kps)
    }

    #[test]
    fn decode_converts_distances_to_a_box() {
        // Stride 8, centre (0,0), distances l=1,t=2,r=3,b=4 in stride units.
        let (s, b, _) = one_cell(0.9, [1.0, 2.0, 3.0, 4.0], None);
        let d = decode_stride(&s, &b, None, 1, 1, 8, 0.5);
        assert_eq!(d.len(), 1);
        assert_eq!(
            (d[0].x1, d[0].y1, d[0].x2, d[0].y2),
            (-8.0, -16.0, 24.0, 32.0)
        );
    }

    #[test]
    fn decode_scales_distances_by_stride() {
        let (s, b, _) = one_cell(0.9, [1.0, 1.0, 1.0, 1.0], None);
        let d8 = decode_stride(&s, &b, None, 1, 1, 8, 0.5);
        let d32 = decode_stride(&s, &b, None, 1, 1, 32, 0.5);
        // Same distances at a coarser stride must yield a 4x larger box.
        assert!((d32[0].width() / d8[0].width() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn decode_respects_score_threshold() {
        let (s, b, _) = one_cell(0.4, [1.0, 1.0, 1.0, 1.0], None);
        assert!(decode_stride(&s, &b, None, 1, 1, 8, 0.5).is_empty());
        assert_eq!(decode_stride(&s, &b, None, 1, 1, 8, 0.3).len(), 1);
    }

    #[test]
    fn decode_rejects_nan_scores() {
        // A corrupt output must not slip through a naive `< threshold` comparison.
        let (s, b, _) = one_cell(f32::NAN, [1.0, 1.0, 1.0, 1.0], None);
        assert!(decode_stride(&s, &b, None, 1, 1, 8, 0.5).is_empty());
    }

    #[test]
    fn decode_landmarks_add_offsets_and_scale_by_stride() {
        let kps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (s, b, k) = one_cell(0.9, [1.0, 1.0, 1.0, 1.0], Some(kps));
        let d = decode_stride(&s, &b, k.as_deref(), 1, 1, 8, 0.5);
        let l = d[0].landmarks.expect("kps head present");
        // centre (0,0) + offset * stride
        assert_eq!(l.left_eye(), (8.0, 16.0));
        assert_eq!(l.right_eye(), (24.0, 32.0));
        assert_eq!(l.points[4], (72.0, 80.0));
    }

    #[test]
    fn decode_without_kps_yields_no_landmarks() {
        let (s, b, _) = one_cell(0.9, [1.0, 1.0, 1.0, 1.0], None);
        let d = decode_stride(&s, &b, None, 1, 1, 8, 0.5);
        assert!(d[0].landmarks.is_none());
    }

    #[test]
    fn decode_truncated_kps_tensor_degrades_to_none_not_panic() {
        // A malformed third-party export must not take down a worker thread.
        let (s, b, _) = one_cell(0.9, [1.0, 1.0, 1.0, 1.0], None);
        let short_kps = vec![0.0f32; 3];
        let d = decode_stride(&s, &b, Some(&short_kps), 1, 1, 8, 0.5);
        assert_eq!(d.len(), 1);
        assert!(d[0].landmarks.is_none());
    }

    #[test]
    fn decode_short_tensors_do_not_panic() {
        // Grid claims 80x80 but tensors hold one prediction.
        let d = decode_stride(&[0.9], &[1.0, 1.0, 1.0, 1.0], None, 80, 80, 8, 0.5);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn decode_empty_tensors_yield_nothing() {
        assert!(decode_stride(&[], &[], None, 80, 80, 8, 0.5).is_empty());
    }

    #[test]
    fn decode_uses_correct_center_for_later_cells() {
        // 2x1 grid at stride 16: anchor index 2 is cell (row0,col1) -> centre (16, 0).
        let scores = vec![0.0, 0.0, 0.9, 0.0];
        // Prediction index 2 occupies bboxes[8..12].
        let mut bboxes = vec![0.0f32; 16];
        bboxes[8] = 1.0;
        bboxes[9] = 1.0;
        bboxes[10] = 1.0;
        bboxes[11] = 1.0;
        let d = decode_stride(&scores, &bboxes, None, 2, 1, 16, 0.5);
        assert_eq!(d.len(), 1);
        // centre (16,0) with unit distances at stride 16 -> [0, -16, 32, 16]
        assert_eq!(
            (d[0].x1, d[0].y1, d[0].x2, d[0].y2),
            (0.0, -16.0, 32.0, 16.0)
        );
    }

    // -- Coordinate mapping -------------------------------------------------

    #[test]
    fn to_original_undoes_letterbox_scale() {
        let lb = Letterbox::compute(1280, 720, 640); // scale 0.5
        let d = FaceDetection {
            x1: 10.0,
            y1: 20.0,
            x2: 30.0,
            y2: 60.0,
            score: 0.9,
            landmarks: Landmarks::from_flat(&[
                10.0, 20.0, 30.0, 20.0, 20.0, 30.0, 12.0, 50.0, 28.0, 50.0,
            ]),
        };
        let o = to_original(d, &lb);
        assert_eq!((o.x1, o.y1, o.x2, o.y2), (20.0, 40.0, 60.0, 120.0));
        assert_eq!(o.landmarks.unwrap().left_eye(), (20.0, 40.0));
    }

    // -- Pre-processing -----------------------------------------------------

    #[test]
    fn preprocess_emits_nchw_of_the_right_length() {
        let img = RgbImage::new(100, 50);
        let (t, lb) = preprocess(&img, 640);
        assert_eq!(t.len(), 3 * 640 * 640);
        assert_eq!(lb.input_size, 640);
    }

    #[test]
    fn preprocess_normalises_to_the_scrfd_range() {
        let mut img = RgbImage::new(8, 8);
        for b in img.as_mut_slice().iter_mut() {
            *b = 255;
        }
        let (t, _) = preprocess(&img, 8);
        // 255 -> (255 - 127.5)/128 = 0.99609375
        assert!((t[0] - 0.996_093_75).abs() < 1e-6);
    }

    #[test]
    fn preprocess_pads_with_normalised_zero_not_grey() {
        // A landscape image into a square leaves bottom rows padded. Those must equal
        // the normalisation of byte 0, matching the reference zero-filled uint8 buffer.
        let mut img = RgbImage::new(16, 8);
        for b in img.as_mut_slice().iter_mut() {
            *b = 200;
        }
        let (t, lb) = preprocess(&img, 16);
        assert_eq!(lb.resized_h, 8);
        let expected_pad = (0.0 - INPUT_MEAN) / INPUT_STD;
        // Last row of the R plane lies in the padded region.
        let idx = 15 * 16;
        assert!((t[idx] - expected_pad).abs() < 1e-6, "got {}", t[idx]);
    }

    #[test]
    fn preprocess_separates_channels_into_planes() {
        let mut img = RgbImage::new(4, 4);
        for i in 0..16 {
            img.as_mut_slice()[i * 3] = 255; // R
            img.as_mut_slice()[i * 3 + 1] = 0; // G
            img.as_mut_slice()[i * 3 + 2] = 128; // B
        }
        let (t, _) = preprocess(&img, 4);
        let plane = 4 * 4;
        assert!(t[0] > 0.9, "R plane");
        assert!((t[plane] - (-0.996_093_75)).abs() < 1e-5, "G plane");
        assert!((t[2 * plane] - (0.5 / 128.0)).abs() < 1e-3, "B plane");
    }

    #[test]
    fn preprocess_zero_size_image_is_all_padding() {
        let img = RgbImage::new(0, 0);
        let (t, _) = preprocess(&img, 8);
        let pad = (0.0 - INPUT_MEAN) / INPUT_STD;
        assert!(t.iter().all(|v| (*v - pad).abs() < 1e-6));
    }

    // -- Output probing -----------------------------------------------------

    #[test]
    fn probe_distinguishes_kps_and_non_kps_exports() {
        assert_eq!(
            probe_outputs(9),
            Some((3, true)),
            "9 outputs = 3 strides with kps"
        );
        assert_eq!(
            probe_outputs(6),
            Some((3, false)),
            "6 outputs = 3 strides, no kps"
        );
    }

    #[test]
    fn probe_rejects_unrecognised_output_counts() {
        // Better to refuse than to mis-slice tensors from an unknown architecture.
        for n in [0, 1, 2, 4, 5, 7, 8, 10, 12] {
            assert!(probe_outputs(n).is_none(), "{n} outputs should be rejected");
        }
    }

    // -- End-to-end post-processing ----------------------------------------

    #[test]
    fn postprocess_merges_strides_and_applies_nms() {
        let cfg = ScrfdConfig::default().with_input_size(640);
        let lb = Letterbox::compute(640, 640, 640);

        // Stride 8 and stride 16 both fire on the same face -> NMS keeps one.
        let g8 = 640 / 8;
        let mut s8 = vec![0.0f32; g8 * g8 * NUM_ANCHORS];
        let mut b8 = vec![0.0f32; g8 * g8 * NUM_ANCHORS * 4];
        s8[0] = 0.9;
        b8[0..4].copy_from_slice(&[0.0, 0.0, 4.0, 4.0]); // 0,0 -> 32,32

        let g16 = 640 / 16;
        let mut s16 = vec![0.0f32; g16 * g16 * NUM_ANCHORS];
        let mut b16 = vec![0.0f32; g16 * g16 * NUM_ANCHORS * 4];
        s16[0] = 0.8;
        b16[0..4].copy_from_slice(&[0.0, 0.0, 2.0, 2.0]); // 0,0 -> 32,32

        let g32 = 640 / 32;
        let s32 = vec![0.0f32; g32 * g32 * NUM_ANCHORS];
        let b32 = vec![0.0f32; g32 * g32 * NUM_ANCHORS * 4];

        let outs = vec![
            StrideOutput {
                scores: &s8,
                bboxes: &b8,
                kps: None,
            },
            StrideOutput {
                scores: &s16,
                bboxes: &b16,
                kps: None,
            },
            StrideOutput {
                scores: &s32,
                bboxes: &b32,
                kps: None,
            },
        ];
        let dets = postprocess(&outs, &lb, 640, 640, &cfg);
        assert_eq!(
            dets.len(),
            1,
            "duplicate cross-stride boxes must be suppressed"
        );
        assert!((dets[0].score - 0.9).abs() < 1e-6, "higher score survives");
    }

    #[test]
    fn postprocess_filters_by_min_face_size() {
        let cfg = ScrfdConfig::default()
            .with_input_size(640)
            .with_min_face_size(100.0);
        let lb = Letterbox::compute(640, 640, 640);
        let g8 = 640 / 8;
        let mut s8 = vec![0.0f32; g8 * g8 * NUM_ANCHORS];
        let mut b8 = vec![0.0f32; g8 * g8 * NUM_ANCHORS * 4];
        s8[0] = 0.9;
        b8[0..4].copy_from_slice(&[0.0, 0.0, 2.0, 2.0]); // 16x16 box, too small

        let outs = vec![StrideOutput {
            scores: &s8,
            bboxes: &b8,
            kps: None,
        }];
        assert!(postprocess(&outs, &lb, 640, 640, &cfg).is_empty());
    }

    #[test]
    fn postprocess_clamps_boxes_into_the_frame() {
        let cfg = ScrfdConfig::default().with_input_size(640);
        let lb = Letterbox::compute(640, 640, 640);
        let g8 = 640 / 8;
        let mut s8 = vec![0.0f32; g8 * g8 * NUM_ANCHORS];
        let mut b8 = vec![0.0f32; g8 * g8 * NUM_ANCHORS * 4];
        s8[0] = 0.9;
        // Large left/top distances push the box off the top-left corner.
        b8[0..4].copy_from_slice(&[5.0, 5.0, 10.0, 10.0]);

        let outs = vec![StrideOutput {
            scores: &s8,
            bboxes: &b8,
            kps: None,
        }];
        let d = postprocess(&outs, &lb, 640, 640, &cfg);
        assert_eq!(d.len(), 1);
        assert!(
            d[0].x1 >= 0.0 && d[0].y1 >= 0.0,
            "box must be clamped to the frame"
        );
    }

    #[test]
    fn postprocess_on_all_zero_scores_finds_nothing() {
        let cfg = ScrfdConfig::default().with_input_size(640);
        let lb = Letterbox::compute(640, 640, 640);
        let g = 640 / 8;
        let s = vec![0.0f32; g * g * NUM_ANCHORS];
        let b = vec![0.0f32; g * g * NUM_ANCHORS * 4];
        let outs = vec![StrideOutput {
            scores: &s,
            bboxes: &b,
            kps: None,
        }];
        assert!(postprocess(&outs, &lb, 640, 640, &cfg).is_empty());
    }

    #[test]
    fn postprocess_ignores_extra_stride_outputs() {
        // More output groups than STRIDES must not index out of bounds.
        let cfg = ScrfdConfig::default().with_input_size(640);
        let lb = Letterbox::compute(640, 640, 640);
        let s = vec![0.0f32; 8];
        let b = vec![0.0f32; 32];
        let outs: Vec<StrideOutput> = (0..5)
            .map(|_| StrideOutput {
                scores: &s,
                bboxes: &b,
                kps: None,
            })
            .collect();
        assert!(postprocess(&outs, &lb, 640, 640, &cfg).is_empty());
    }

    #[test]
    fn scrfd_constants_match_published_preprocessing() {
        assert_eq!(INPUT_MEAN, 127.5);
        assert_eq!(
            INPUT_STD, 128.0,
            "SCRFD uses 1/128, unlike ArcFace's 1/127.5"
        );
        assert_eq!(STRIDES, [8, 16, 32]);
        assert_eq!(NUM_ANCHORS, 2);
    }
}
