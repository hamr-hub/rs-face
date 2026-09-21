//! Lightweight 5-point facial landmark estimator.
//!
//! Given an axis-aligned face bounding box and a grayscale frame, estimate
//! the five canonical facial keypoints in image-pixel coordinates:
//!
//!   left eye center, right eye center, nose tip,
//!   left mouth corner, right mouth corner.
//!
//! The ordering matches [`crate::face::landmark`], so
//! [`FivePointLandmarks::to_face_landmarks`] is a zero-cost conversion into
//! the [`crate::face::Landmarks`] consumed by
//! [`crate::align::estimate_arcface_transform`] / [`crate::align::norm_crop`].
//!
//! ## Algorithm (zero-dep, ~no learned weights)
//!
//! Each landmark is located by a *column/row projection* in a region where
//! anatomy places it. Specifically:
//!
//! - **Eyes**: the upper half of the face is split into a left and a right
//!   sub-band; in each, the algorithm sweeps every column and picks the
//!   darkest one by mean brightness over a `[0.30h .. 0.50h]` vertical band,
//!   then refines the y-coordinate as the darkest row inside that column.
//! - **Nose**: a vertical sub-strip at horizontal centre, `[0.40w .. 0.60w]`
//!   × `[0.45h .. 0.65h]`, gets the same darkness sweep; the result is the
//!   tip of the nasal ridge.
//! - **Mouth corners**: the lower band `[0.65h .. 0.85h]` is row-projected
//!   to find the darkest row (the mouth line); on that row, the leftmost
//!   and rightmost pixels darker than `(row_mean − 20)` are taken as the
//!   two corners.
//!
//! This is "Option A" from the spec: pure classical CV, <1 ms per face on a
//! typical photo, no model file required. It is meant to be a usable
//! geometric baseline — accurate enough for *alignment seeding* and demo
//! overlays, not a replacement for a learned regressor on difficult poses
//! (extreme yaw, occlusion, profile views). For those, use a SCRFD /
//! RetinaFace / YuNet model via [`crate::scrfd_detector`] which already
//! produces sub-pixel landmarks.
//!
//! ## Why no `crate::integral`?
//!
//! The integral-image machinery is feature-gated under `detector-haar`; if
//! the landmarks module pulled it in, the zero-dep `cargo build
//! --no-default-features --lib` would no longer satisfy this module.
//! All region-mean computations here are done with the bare inner loop
//! (O(w·h) per band, but the bands are tiny fractions of the bbox).

use crate::face::{Detection, Landmarks, NUM_LANDMARKS};
use crate::image::GrayImage;

/// Axis-aligned integer bounding box used by [`estimate_five_point`].
///
/// Defined locally so the module compiles without the detector stack. The
/// four fields match the convention of [`Detection`], and a `From` impl is
/// provided so a fresh detector hit can be passed straight in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    /// Build a `Rect` from an existing [`Detection`].
    pub fn from_detection(d: &Detection) -> Self {
        Self {
            x: d.x,
            y: d.y,
            w: d.w,
            h: d.h,
        }
    }

    /// Right edge, exclusive.
    #[inline]
    pub fn right(&self) -> usize {
        self.x + self.w
    }

    /// Bottom edge, exclusive.
    #[inline]
    pub fn bottom(&self) -> usize {
        self.y + self.h
    }

    /// Clamp to `[0, w) × [0, h)` so all subsequent indexing is in-range.
    /// Saturation to `0` for any field is a deliberate guard: an empty
    /// `Rect` returns the centre-based fallback in [`estimate_five_point`].
    pub fn clamp_to(&self, img_w: usize, img_h: usize) -> Self {
        if self.w == 0 || self.h == 0 {
            return Self {
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            };
        }
        let x = self.x.min(img_w);
        let y = self.y.min(img_h);
        let x2 = self.right().min(img_w);
        let y2 = self.bottom().min(img_h);
        Self {
            x,
            y,
            w: x2.saturating_sub(x),
            h: y2.saturating_sub(y),
        }
    }
}

impl From<Detection> for Rect {
    fn from(d: Detection) -> Self {
        Self::from_detection(&d)
    }
}

/// Five canonical facial landmarks in image-pixel coordinates.
///
/// Ordering matches [`crate::face::landmark`]:
/// `LEFT_EYE, RIGHT_EYE, NOSE, LEFT_MOUTH, RIGHT_MOUTH`
/// (i.e. InsightFace order, where "left" is the viewer's left).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FivePointLandmarks {
    pub left_eye: (f32, f32),
    pub right_eye: (f32, f32),
    pub nose: (f32, f32),
    pub left_mouth: (f32, f32),
    pub right_mouth: (f32, f32),
}

impl FivePointLandmarks {
    /// Inter-ocular distance in pixels (Euclidean between the two eye
    /// centres). Cheap pose-robust size proxy — useful as a quality gate
    /// before any downstream alignment or embedding.
    pub fn eye_distance(&self) -> f32 {
        let (lx, ly) = self.left_eye;
        let (rx, ry) = self.right_eye;
        ((rx - lx).powi(2) + (ry - ly).powi(2)).sqrt()
    }

    /// Pack into the [`Landmarks`] array shape consumed by
    /// [`crate::align::estimate_arcface_transform`] and
    /// [`crate::align::norm_crop`].
    pub fn to_face_landmarks(self) -> Landmarks {
        Landmarks {
            points: [
                self.left_eye,
                self.right_eye,
                self.nose,
                self.left_mouth,
                self.right_mouth,
            ],
        }
    }

    /// Build from the canonical [`Landmarks`] array.
    pub fn from_face_landmarks(l: &Landmarks) -> Self {
        Self {
            left_eye: l.points[crate::face::landmark::LEFT_EYE],
            right_eye: l.points[crate::face::landmark::RIGHT_EYE],
            nose: l.points[crate::face::landmark::NOSE],
            left_mouth: l.points[crate::face::landmark::LEFT_MOUTH],
            right_mouth: l.points[crate::face::landmark::RIGHT_MOUTH],
        }
    }
}

/// Estimate five facial landmarks inside `bbox` from a grayscale frame.
///
/// `bbox` is clamped to the image bounds internally; callers may pass a
/// slightly off-frame box (e.g. an edge-touching detection) without
/// crashing. A degenerate or empty bbox falls back to a centre-based
/// anatomical guess so downstream alignment still gets plausible points
/// rather than a NaN-poisoned crop.
///
/// Cost: O(w·h) over a few narrow bands — well under 1 ms for any normal
/// detection on a multi-MP frame.
pub fn estimate_five_point(gray: &GrayImage, bbox: Rect) -> FivePointLandmarks {
    let bbox = bbox.clamp_to(gray.width(), gray.height());

    // Degenerate bbox: return an anatomical guess at the image centre so
    // the consumer still has something to render / align against, rather
    // than silently producing NaN landmarks.
    if bbox.w < 4 || bbox.h < 4 {
        let cx = gray.width() as f32 / 2.0;
        let cy = gray.height() as f32 / 2.0;
        return FivePointLandmarks {
            left_eye: (cx - 6.0, cy - 4.0),
            right_eye: (cx + 6.0, cy - 4.0),
            nose: (cx, cy),
            left_mouth: (cx - 8.0, cy + 8.0),
            right_mouth: (cx + 8.0, cy + 8.0),
        };
    }

    let (left_eye, right_eye) = locate_eyes(gray, &bbox);
    let nose = locate_nose(gray, &bbox);
    let (left_mouth, right_mouth) = locate_mouth_corners(gray, &bbox);

    FivePointLandmarks {
        left_eye,
        right_eye,
        nose,
        left_mouth,
        right_mouth,
    }
}

/// Find the left and right eye centres by column-projection on the
/// upper-half band `[0.30h, 0.50h]`.
fn locate_eyes(gray: &GrayImage, bbox: &Rect) -> ((f32, f32), (f32, f32)) {
    let y_start = bbox.y + bbox.h * 3 / 10;
    let y_end = bbox.y + bbox.h / 2;
    if y_end <= y_start {
        let cy = (y_start + y_end) as f32 / 2.0;
        return (
            (bbox.x as f32 + bbox.w as f32 * 0.30, cy),
            (bbox.x as f32 + bbox.w as f32 * 0.70, cy),
        );
    }
    let half_w = bbox.w / 2;

    let left = darkest_column_in_band(gray, bbox.x, bbox.x + half_w, y_start, y_end);
    let right = darkest_column_in_band(gray, bbox.x + half_w, bbox.right(), y_start, y_end);

    (
        (left.0 as f32 + 0.5, left.1 as f32 + 0.5),
        (right.0 as f32 + 0.5, right.1 as f32 + 0.5),
    )
}

/// Inside the rectangle `[x_lo, x_hi) × [y_lo, y_hi)`, find the column
/// whose mean luminance is darkest, then refine the row as the darkest
/// single pixel within that column. Returns integer pixel coordinates.
fn darkest_column_in_band(
    gray: &GrayImage,
    x_lo: usize,
    x_hi: usize,
    y_lo: usize,
    y_hi: usize,
) -> (usize, usize) {
    let band_h = (y_hi - y_lo).max(1);
    let mut best_x = (x_lo + x_hi) / 2;
    let mut best_mean = f32::MAX;
    for x in x_lo..x_hi {
        let mut sum: u32 = 0;
        for y in y_lo..y_hi {
            sum += gray[(x, y)] as u32;
        }
        let mean = sum as f32 / band_h as f32;
        if mean < best_mean {
            best_mean = mean;
            best_x = x;
        }
    }
    // Refine y as the darkest row in this column.
    let mut best_y = (y_lo + y_hi) / 2;
    let mut best_v = f32::MAX;
    for y in y_lo..y_hi {
        let v = gray[(best_x, y)] as f32;
        if v < best_v {
            best_v = v;
            best_y = y;
        }
    }
    (best_x, best_y)
}

/// Locate the nose tip as the darkest column-then-row inside the central
/// `[0.40w, 0.60w] × [0.45h, 0.65h]` band.
fn locate_nose(gray: &GrayImage, bbox: &Rect) -> (f32, f32) {
    let x_lo = bbox.x + bbox.w * 40 / 100;
    let x_hi = bbox.x + bbox.w * 60 / 100;
    let y_lo = bbox.y + bbox.h * 45 / 100;
    let y_hi = bbox.y + bbox.h * 65 / 100;
    if x_hi <= x_lo || y_hi <= y_lo {
        let cx = (x_lo + x_hi) as f32 / 2.0;
        let cy = (y_lo + y_hi) as f32 / 2.0;
        return (cx, cy);
    }
    let (x, y) = darkest_column_in_band(gray, x_lo, x_hi, y_lo, y_hi);
    (x as f32 + 0.5, y as f32 + 0.5)
}

/// Locate the mouth corners by row-projection in the lower band
/// `[0.65h, 0.85h]`. Picks the darkest row, then takes the leftmost and
/// rightmost pixels darker than `row_mean − 20` on that row.
fn locate_mouth_corners(gray: &GrayImage, bbox: &Rect) -> ((f32, f32), (f32, f32)) {
    let y_lo = bbox.y + bbox.h * 65 / 100;
    let y_hi = bbox.y + bbox.h * 85 / 100;
    let x_lo = bbox.x + bbox.w / 4;
    let x_hi = bbox.right() - bbox.w / 4;

    if y_hi <= y_lo || x_hi <= x_lo {
        let cx = bbox.x as f32 + bbox.w as f32 / 2.0;
        let cy = (y_lo + y_hi) as f32 / 2.0;
        return (
            (cx - bbox.w as f32 * 0.20, cy),
            (cx + bbox.w as f32 * 0.20, cy),
        );
    }

    // Darkest row in the band.
    let mut best_y = (y_lo + y_hi) / 2;
    let mut best_row_mean = f32::MAX;
    for y in y_lo..y_hi {
        let mut sum: u32 = 0;
        for x in bbox.x..bbox.right() {
            sum += gray[(x, y)] as u32;
        }
        let mean = sum as f32 / bbox.w as f32;
        if mean < best_row_mean {
            best_row_mean = mean;
            best_y = y;
        }
    }

    // Mean of the chosen row over the whole bbox width — used as the
    // threshold baseline so the algorithm doesn't need a hand-tuned
    // brightness constant.
    let row_mean: f32 = {
        let mut sum: u32 = 0;
        for x in bbox.x..bbox.right() {
            sum += gray[(x, best_y)] as u32;
        }
        sum as f32 / bbox.w as f32
    };
    let threshold = row_mean - 20.0;

    // Leftmost / rightmost "dark" pixel on that row, restricted to the
    // mouth's horizontal range so cheek shadows can't steal the corners.
    let mut left_x = x_lo;
    let mut found_left = false;
    for x in x_lo..x_hi {
        if gray[(x, best_y)] as f32 <= threshold {
            left_x = x;
            found_left = true;
            break;
        }
    }
    let mut right_x = x_hi;
    let mut found_right = false;
    for x in (x_lo..x_hi).rev() {
        if gray[(x, best_y)] as f32 <= threshold {
            right_x = x;
            found_right = true;
            break;
        }
    }

    if !found_left || !found_right || right_x <= left_x {
        // No clear mouth line in the band — fall back to symmetric
        // anatomical guess on the chosen row.
        let cx = bbox.x as f32 + bbox.w as f32 / 2.0;
        return (
            (cx - bbox.w as f32 * 0.20, best_y as f32 + 0.5),
            (cx + bbox.w as f32 * 0.20, best_y as f32 + 0.5),
        );
    }

    (
        (left_x as f32 + 0.5, best_y as f32 + 0.5),
        (right_x as f32 + 0.5, best_y as f32 + 0.5),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 64×64 synthetic frontal face with five explicitly placed
    /// landmarks and return both the image and the ground-truth positions.
    ///
    /// Skin tone: 180. Eyes are 5×5 dark patches at `(20, 25)` and
    /// `(44, 25)`. Nose is a 3×5 dark patch at `(32, 35)`. Mouth is a
    /// horizontal dark stripe at `y = 46` running from `x = 22` to `x = 42`.
    fn synthetic_face() -> (GrayImage, [(f32, f32); NUM_LANDMARKS]) {
        let mut img = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                img[(x, y)] = 180;
            }
        }
        // Left eye: 5x5 dark patch centred on (20, 25).
        for dy in 0..5 {
            for dx in 0..5 {
                img[(18 + dx, 23 + dy)] = 50;
            }
        }
        // Right eye.
        for dy in 0..5 {
            for dx in 0..5 {
                img[(42 + dx, 23 + dy)] = 50;
            }
        }
        // Nose: 3-wide, 5-tall dark vertical patch centred on (32, 35).
        for dy in 0..5 {
            for dx in 0..3 {
                img[(31 + dx, 33 + dy)] = 80;
            }
        }
        // Mouth: horizontal dark line at y=46, x in [22, 42].
        for x in 22..=42 {
            img[(x, 46)] = 60;
        }

        let gt = [
            (20.0, 25.0), // left eye
            (44.0, 25.0), // right eye
            (32.0, 35.0), // nose
            (22.0, 46.0), // left mouth
            (42.0, 46.0), // right mouth
        ];
        (img, gt)
    }

    /// The headline test from the task spec: estimator must hit every
    /// landmark within 3 px of ground truth on a clean synthetic face.
    #[test]
    fn estimate_five_point_synthetic_face() {
        let (img, gt) = synthetic_face();
        let bbox = Rect {
            x: 0,
            y: 0,
            w: 64,
            h: 64,
        };
        let lms = estimate_five_point(&img, bbox);

        let tol = 3.0_f32;
        let cases: [(&str, (f32, f32), (f32, f32)); 5] = [
            ("left_eye", lms.left_eye, gt[0]),
            ("right_eye", lms.right_eye, gt[1]),
            ("nose", lms.nose, gt[2]),
            ("left_mouth", lms.left_mouth, gt[3]),
            ("right_mouth", lms.right_mouth, gt[4]),
        ];
        for (name, est, truth) in cases {
            let dx = (est.0 - truth.0).abs();
            let dy = (est.1 - truth.1).abs();
            assert!(
                dx <= tol && dy <= tol,
                "{name}: estimated=({:.2},{:.2}) ground_truth=({:.2},{:.2}) dx={:.2} dy={:.2} tol={tol}",
                est.0,
                est.1,
                truth.0,
                truth.1,
                dx,
                dy,
            );
        }
    }

    /// Degenerate bbox must not panic and must return finite values.
    #[test]
    fn degenerate_bbox_returns_anatomical_guess() {
        let img = GrayImage::new(20, 20);
        let lms = estimate_five_point(
            &img,
            Rect {
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            },
        );
        for p in [
            lms.left_eye,
            lms.right_eye,
            lms.nose,
            lms.left_mouth,
            lms.right_mouth,
        ] {
            assert!(p.0.is_finite() && p.1.is_finite(), "{:?} not finite", p);
        }
    }

    /// Off-frame bbox must be clamped internally — no out-of-bounds read.
    #[test]
    fn off_frame_bbox_is_clamped_safely() {
        let img = GrayImage::new(64, 64);
        let lms = estimate_five_point(
            &img,
            Rect {
                x: 60,
                y: 60,
                w: 100,
                h: 100,
            },
        );
        for p in [
            lms.left_eye,
            lms.right_eye,
            lms.nose,
            lms.left_mouth,
            lms.right_mouth,
        ] {
            assert!(p.0.is_finite() && p.1.is_finite());
            assert!(
                (p.0 - 32.0).abs() < 64.0 && (p.1 - 32.0).abs() < 64.0,
                "landmark out of frame: {p:?}"
            );
        }
    }

    /// `Rect::from_detection` must round-trip pixel-perfect.
    #[test]
    fn rect_from_detection_roundtrips() {
        let d = Detection {
            x: 4,
            y: 6,
            w: 20,
            h: 24,
            score: 0.9,
        };
        let r = Rect::from_detection(&d);
        assert_eq!((r.x, r.y, r.w, r.h), (4, 6, 20, 24));
        assert_eq!(r.right(), 24);
        assert_eq!(r.bottom(), 30);
    }

    /// `to_face_landmarks` must preserve the InsightFace landmark order.
    #[test]
    fn to_face_landmarks_preserves_insightface_order() {
        let lms = FivePointLandmarks {
            left_eye: (1.0, 2.0),
            right_eye: (3.0, 4.0),
            nose: (5.0, 6.0),
            left_mouth: (7.0, 8.0),
            right_mouth: (9.0, 10.0),
        };
        let fl = lms.to_face_landmarks();
        assert_eq!(fl.left_eye(), (1.0, 2.0));
        assert_eq!(fl.right_eye(), (3.0, 4.0));
        assert_eq!(fl.nose(), (5.0, 6.0));
        assert_eq!(fl.points[crate::face::landmark::LEFT_MOUTH], (7.0, 8.0));
        assert_eq!(fl.points[crate::face::landmark::RIGHT_MOUTH], (9.0, 10.0));
    }

    /// eye_distance is the Euclidean between the two eye centres.
    #[test]
    fn eye_distance_is_euclidean() {
        let lms = FivePointLandmarks {
            left_eye: (0.0, 0.0),
            right_eye: (3.0, 4.0),
            nose: (0.0, 0.0),
            left_mouth: (0.0, 0.0),
            right_mouth: (0.0, 0.0),
        };
        assert!((lms.eye_distance() - 5.0).abs() < 1e-6);
    }

    /// clamp_to saturates a wholly-out-of-image bbox to (0, 0).
    #[test]
    fn rect_clamp_to_saturates_outside_image() {
        let r = Rect {
            x: 100,
            y: 200,
            w: 50,
            h: 60,
        };
        let c = r.clamp_to(64, 64);
        assert_eq!(c.x, 64);
        assert_eq!(c.y, 64);
        assert_eq!(c.w, 0);
        assert_eq!(c.h, 0);
    }
}
