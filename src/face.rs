//! Detection value types shared by every detector: the classical integer box
//! ([`Detection`] + greedy [`non_max_suppression`]) and the modern sub-pixel
//! box with 5-point landmarks ([`FaceDetection`] + [`nms`]).
//!
//! The classical [`Detection`] is pixel-space and carries no landmarks, which
//! is all a Haar cascade can produce. Modern single-stage detectors (SCRFD,
//! RetinaFace, YuNet) regress *continuous* box offsets and five keypoints, and
//! downstream face **recognition** is critically dependent on those keypoints:
//! ArcFace embeddings are only comparable when every crop has been warped onto
//! the same canonical landmark configuration.
//!
//! Rounding to `usize` at detect time would throw away the sub-pixel precision that
//! the similarity transform in the ONNX-only `crate::align` module needs, so
//! [`FaceDetection`] keeps `f32` throughout and only quantises at the
//! drawing/serialisation boundary.

/// A single detection: pixel-space bounding box + confidence score.
///
/// This is the classical output type of every zero-dep detector; it lives in
/// the `face` module (rather than next to the Haar scanner) so that
/// landmark-based detectors can produce it without depending on the Haar
/// detector machinery.
#[derive(Clone, Debug)]
pub struct Detection {
    /// Left edge in pixels.
    pub x: usize,
    /// Top edge in pixels.
    pub y: usize,
    /// Width in pixels.
    pub w: usize,
    /// Height in pixels.
    pub h: usize,
    /// Detector confidence (cascade stage score or network sigmoid).
    pub score: f32,
}

impl Detection {
    /// Right edge (exclusive) of the bounding box.
    #[inline]
    pub fn right(&self) -> usize {
        self.x + self.w
    }

    /// Bottom edge (exclusive) of the bounding box.
    #[inline]
    pub fn bottom(&self) -> usize {
        self.y + self.h
    }

    /// Area in pixels (`w * h`).
    #[inline]
    pub fn area(&self) -> usize {
        self.w * self.h
    }

    /// Intersection-over-union with another detection, in `[0, 1]`.
    /// Overlapping boxes return `> 0`; disjoint boxes return exactly `0`.
    pub fn iou(&self, other: &Detection) -> f32 {
        iou(self, other)
    }

    /// Center point `(cx, cy)` of the bounding box.
    #[inline]
    pub fn center(&self) -> (f32, f32) {
        (
            self.x as f32 + self.w as f32 / 2.0,
            self.y as f32 + self.h as f32 / 2.0,
        )
    }
}

/// Standard greedy NMS over integer boxes: highest score first, suppress all
/// with IoU above the threshold.
///
/// Kept separate from [`nms`] (sub-pixel boxes): reusing one function for both
/// would force a rounding pass whose ties change which near-identical box wins.
pub fn non_max_suppression(mut dets: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep: Vec<Detection> = Vec::new();
    let mut suppressed = vec![false; dets.len()];
    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if suppressed[j] {
                continue;
            }
            if iou(&dets[i], &dets[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

#[inline]
pub(crate) fn iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let w = (x2 as i64 - x1 as i64).max(0) as usize;
    let h = (y2 as i64 - y1 as i64).max(0) as usize;
    let inter = (w * h) as f32;
    if inter <= 0.0 {
        return 0.0;
    }
    let union = (a.w * a.h + b.w * b.h) as f32 - inter;
    if union <= 0.0 {
        return 0.0;
    }
    inter / union
}

/// Canonical index of each of the five landmarks, in InsightFace order.
///
/// This ordering is a hard ABI contract with the ArcFace reference landmarks
/// (`crate::align::ARCFACE_REFERENCE_LANDMARKS`, ONNX backend only) — permuting
/// it silently destroys recognition accuracy rather than producing an error, so
/// it is spelled out as named constants instead of bare indices.
pub mod landmark {
    pub const LEFT_EYE: usize = 0;
    pub const RIGHT_EYE: usize = 1;
    pub const NOSE: usize = 2;
    pub const LEFT_MOUTH: usize = 3;
    pub const RIGHT_MOUTH: usize = 4;
}

/// Number of landmarks produced by the SCRFD / RetinaFace / YuNet family.
pub const NUM_LANDMARKS: usize = 5;

/// Five facial keypoints in image pixel coordinates, `[(x, y); 5]`.
///
/// Ordering is given by [`landmark`]. "Left" means the subject's left as it appears
/// on the *image's* left-hand side (i.e. the viewer's left), matching InsightFace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landmarks {
    pub points: [(f32, f32); NUM_LANDMARKS],
}

impl Landmarks {
    /// Build from a flat `[x0, y0, x1, y1, ...]` slice as emitted by an ONNX kps head.
    ///
    /// Returns `None` on a short slice rather than panicking, because the length here
    /// is determined by a model file the user supplied — a malformed third-party
    /// `.onnx` must surface as a handled error, not a crash inside a worker thread.
    pub fn from_flat(v: &[f32]) -> Option<Self> {
        if v.len() < NUM_LANDMARKS * 2 {
            return None;
        }
        let mut points = [(0.0f32, 0.0f32); NUM_LANDMARKS];
        for (i, p) in points.iter_mut().enumerate() {
            *p = (v[i * 2], v[i * 2 + 1]);
        }
        Some(Self { points })
    }

    #[inline]
    pub fn left_eye(&self) -> (f32, f32) {
        self.points[landmark::LEFT_EYE]
    }

    #[inline]
    pub fn right_eye(&self) -> (f32, f32) {
        self.points[landmark::RIGHT_EYE]
    }

    #[inline]
    pub fn nose(&self) -> (f32, f32) {
        self.points[landmark::NOSE]
    }

    /// Translate every point by `(dx, dy)`.
    ///
    /// Used to undo letterbox padding after inference.
    #[inline]
    pub fn translated(mut self, dx: f32, dy: f32) -> Self {
        for p in self.points.iter_mut() {
            p.0 += dx;
            p.1 += dy;
        }
        self
    }

    /// Scale every point about the origin by `s`.
    ///
    /// Used to map coordinates from the network's input resolution back to the
    /// original frame after a letterbox resize.
    #[inline]
    pub fn scaled(mut self, s: f32) -> Self {
        for p in self.points.iter_mut() {
            p.0 *= s;
            p.1 *= s;
        }
        self
    }

    /// Inter-ocular distance in pixels — a cheap, pose-robust proxy for face size,
    /// useful as a quality gate before spending an embedding forward pass on a crop.
    pub fn eye_distance(&self) -> f32 {
        let (lx, ly) = self.left_eye();
        let (rx, ry) = self.right_eye();
        ((rx - lx).powi(2) + (ry - ly).powi(2)).sqrt()
    }
}

/// A detection from a modern detector: sub-pixel box, score, and optional landmarks.
///
/// Landmarks are `Option` because not every model has a keypoint head (plain SCRFD
/// `_bnkps`-less variants do not), and because the recognition pipeline must be able
/// to *detect* the absence and skip alignment rather than silently align to garbage.
#[derive(Clone, Copy, Debug)]
pub struct FaceDetection {
    /// Left edge in pixels. May be slightly negative before clamping: box regression
    /// is unconstrained and legitimately predicts faces running off the frame edge.
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub landmarks: Option<Landmarks>,
}

impl FaceDetection {
    #[inline]
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    #[inline]
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    #[inline]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    #[inline]
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    /// Intersection-over-union, computed in `f32` so NMS is not biased by the
    /// integer rounding that [`Detection::iou`] necessarily performs.
    pub fn iou(&self, other: &FaceDetection) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        let iw = (ix2 - ix1).max(0.0);
        let ih = (iy2 - iy1).max(0.0);
        let inter = iw * ih;
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// Clamp the box to `[0, w) x [0, h)`.
    ///
    /// Applied only at the very end of decoding: clamping *before* NMS would distort
    /// IoU between two edge faces and cause spurious suppression.
    pub fn clamped(mut self, w: f32, h: f32) -> Self {
        self.x1 = self.x1.clamp(0.0, w);
        self.y1 = self.y1.clamp(0.0, h);
        self.x2 = self.x2.clamp(0.0, w);
        self.y2 = self.y2.clamp(0.0, h);
        self
    }

    /// Expand (or contract, for `factor < 1`) the box about its centre.
    ///
    /// Recognition backbones are trained on crops with a margin; feeding a tight
    /// detector box degrades embedding quality.
    pub fn scaled_about_center(mut self, factor: f32) -> Self {
        let (cx, cy) = self.center();
        let hw = self.width() * 0.5 * factor;
        let hh = self.height() * 0.5 * factor;
        self.x1 = cx - hw;
        self.x2 = cx + hw;
        self.y1 = cy - hh;
        self.y2 = cy + hh;
        self
    }

    /// Lossy conversion to the classical integer [`Detection`] for reuse of the
    /// existing PNG annotator, JSON manifest writer and GPU-parity comparison paths.
    ///
    /// Landmarks are dropped, so this is deliberately *not* a `From` impl — losing
    /// keypoints should be an explicit, visible call at the boundary.
    pub fn to_detection(self) -> Detection {
        let x = self.x1.max(0.0);
        let y = self.y1.max(0.0);
        Detection {
            x: x as usize,
            y: y as usize,
            w: self.width().round().max(0.0) as usize,
            h: self.height().round().max(0.0) as usize,
            score: self.score,
        }
    }
}

impl From<Detection> for FaceDetection {
    /// Widen a classical detection. Landmarks are `None` — a Haar cascade has no
    /// keypoint head, and fabricating plausible-looking eye positions here would
    /// let unusable crops flow into the recognition path.
    fn from(d: Detection) -> Self {
        Self {
            x1: d.x as f32,
            y1: d.y as f32,
            x2: (d.x + d.w) as f32,
            y2: (d.y + d.h) as f32,
            score: d.score,
            landmarks: None,
        }
    }
}

/// Greedy NMS over sub-pixel boxes, highest score first.
///
/// Separate from [`non_max_suppression`] because that function operates on
/// integer boxes; reusing one for both would force a rounding pass whose ties
/// change which of two near-identical boxes survives.
pub fn nms(mut dets: Vec<FaceDetection>, iou_threshold: f32) -> Vec<FaceDetection> {
    // Descending score. `total_cmp` avoids the partial-ord unwrap panic a NaN score
    // from a corrupt model output would otherwise trigger.
    dets.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

    let mut keep: Vec<FaceDetection> = Vec::with_capacity(dets.len());
    'outer: for d in dets {
        for k in &keep {
            if d.iou(k) > iou_threshold {
                continue 'outer;
            }
        }
        keep.push(d);
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> FaceDetection {
        FaceDetection {
            x1,
            y1,
            x2,
            y2,
            score,
            landmarks: None,
        }
    }

    #[test]
    fn iou_identical_boxes_is_one() {
        let a = det(0.0, 0.0, 10.0, 10.0, 1.0);
        assert!((a.iou(&a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_disjoint_boxes_is_zero() {
        let a = det(0.0, 0.0, 10.0, 10.0, 1.0);
        let b = det(20.0, 20.0, 30.0, 30.0, 1.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn iou_half_overlap() {
        // Two 10x10 boxes sharing a 5x10 strip: inter=50, union=150 -> 1/3.
        let a = det(0.0, 0.0, 10.0, 10.0, 1.0);
        let b = det(5.0, 0.0, 15.0, 10.0, 1.0);
        assert!((a.iou(&b) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn iou_nested_box() {
        // Inner fully contained in outer: IoU = area(inner)/area(outer).
        let outer = det(0.0, 0.0, 100.0, 100.0, 1.0);
        let inner = det(10.0, 10.0, 30.0, 30.0, 1.0);
        // inner is 20x20=400, outer is 100x100=10000, IoU=0.04.
        assert!((inner.iou(&outer) - 0.04).abs() < 1e-6);
        assert!((outer.iou(&inner) - 0.04).abs() < 1e-6);
    }

    #[test]
    fn iou_is_symmetric() {
        // a.iou(b) == b.iou(a) across disjoint, partial-overlap, and nested cases.
        let a = det(0.0, 0.0, 10.0, 10.0, 1.0);
        let b = det(5.0, 0.0, 15.0, 10.0, 1.0);
        assert!((a.iou(&b) - b.iou(&a)).abs() < 1e-6);

        let c = det(50.0, 50.0, 60.0, 60.0, 1.0);
        assert!((a.iou(&c) - c.iou(&a)).abs() < 1e-6);

        let inner = det(2.0, 2.0, 8.0, 8.0, 1.0);
        assert!((a.iou(&inner) - inner.iou(&a)).abs() < 1e-6);
    }

    #[test]
    fn nms_suppresses_overlapping_keeps_highest() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0.9),
            det(1.0, 1.0, 11.0, 11.0, 0.95), // heavy overlap, higher score
            det(50.0, 50.0, 60.0, 60.0, 0.5), // disjoint, must survive
        ];
        let kept = nms(dets, 0.5);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].score, 0.95);
        assert_eq!(kept[1].score, 0.5);
    }

    #[test]
    fn nms_empty_input_is_empty_not_panic() {
        assert!(nms(Vec::new(), 0.5).is_empty());
    }

    #[test]
    fn nms_nan_score_does_not_panic() {
        // A corrupt model can emit NaN; total_cmp must keep sort total.
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, f32::NAN),
            det(50.0, 50.0, 60.0, 60.0, 0.5),
        ];
        let kept = nms(dets, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn landmarks_from_flat_rejects_short_slice() {
        assert!(Landmarks::from_flat(&[0.0; 9]).is_none());
        assert!(Landmarks::from_flat(&[0.0; 10]).is_some());
    }

    #[test]
    fn landmarks_roundtrip_order_is_insightface() {
        let flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let l = Landmarks::from_flat(&flat).unwrap();
        assert_eq!(l.left_eye(), (1.0, 2.0));
        assert_eq!(l.right_eye(), (3.0, 4.0));
        assert_eq!(l.nose(), (5.0, 6.0));
        assert_eq!(l.points[landmark::LEFT_MOUTH], (7.0, 8.0));
        assert_eq!(l.points[landmark::RIGHT_MOUTH], (9.0, 10.0));
    }

    #[test]
    fn landmarks_eye_distance() {
        let l = Landmarks::from_flat(&[0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        assert!((l.eye_distance() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn letterbox_undo_is_translate_then_scale() {
        let l = Landmarks::from_flat(&[10.0, 10.0, 20.0, 10.0, 15.0, 15.0, 12.0, 20.0, 18.0, 20.0])
            .unwrap();
        let out = l.translated(-5.0, -2.0).scaled(2.0);
        assert_eq!(out.left_eye(), (10.0, 16.0));
    }

    #[test]
    fn clamped_keeps_box_inside_frame() {
        let d = det(-5.0, -5.0, 105.0, 50.0, 1.0).clamped(100.0, 100.0);
        assert_eq!((d.x1, d.y1, d.x2, d.y2), (0.0, 0.0, 100.0, 50.0));
    }

    #[test]
    fn scaled_about_center_preserves_center() {
        let d = det(10.0, 10.0, 30.0, 30.0, 1.0);
        let s = d.scaled_about_center(1.5);
        assert_eq!(d.center(), s.center());
        assert!((s.width() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn detection_roundtrip_widens_without_landmarks() {
        let d = Detection {
            x: 4,
            y: 6,
            w: 20,
            h: 24,
            score: 0.8,
        };
        let f = FaceDetection::from(d.clone());
        assert!(f.landmarks.is_none());
        let back = f.to_detection();
        assert_eq!((back.x, back.y, back.w, back.h), (d.x, d.y, d.w, d.h));
    }

    #[test]
    fn to_detection_clamps_negative_origin() {
        // Box regression can predict off-frame origins; usize conversion must not wrap.
        let f = det(-3.0, -7.0, 10.0, 10.0, 0.9);
        let d = f.to_detection();
        assert_eq!((d.x, d.y), (0, 0));
    }

    #[test]
    fn classical_detection_helpers() {
        let a = Detection {
            x: 10,
            y: 20,
            w: 30,
            h: 40,
            score: 0.75,
        };
        assert_eq!(a.right(), 40);
        assert_eq!(a.bottom(), 60);
        assert_eq!(a.area(), 1200);
        assert_eq!(a.center(), (25.0, 40.0));
        let b = a.clone();
        assert!((a.iou(&b) - 1.0).abs() < 1e-6);
        let c = Detection {
            x: 200,
            y: 200,
            w: 10,
            h: 10,
            score: 0.1,
        };
        assert_eq!(a.iou(&c), 0.0);
    }

    #[test]
    fn classical_nms_merges_overlapping_keeps_disjoint() {
        let a = Detection {
            x: 0,
            y: 0,
            w: 20,
            h: 20,
            score: 1.0,
        };
        let b = Detection {
            x: 2,
            y: 2,
            w: 20,
            h: 20,
            score: 0.9,
        };
        let c = Detection {
            x: 100,
            y: 100,
            w: 20,
            h: 20,
            score: 0.5,
        };
        let r = non_max_suppression(vec![a, b, c], 0.3);
        assert_eq!(r.len(), 2, "should merge a+b but keep c");
        assert_eq!(r[0].score, 1.0);
    }

    #[test]
    fn classical_nms_empty_input_is_empty() {
        assert!(non_max_suppression(Vec::new(), 0.3).is_empty());
    }
}
