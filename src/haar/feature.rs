//! Haar-like feature primitives.
//!
//! Each feature is a layout of rectangles with weights. Evaluation:
//!   `response = sum(weight_i * rect_sum_i)`
//!
//! We use 5 feature kinds matching OpenCV's classical cascade:
//!  - `VerticalEdge`        : two equal horizontal rectangles stacked.
//!  - `HorizontalEdge`      : two equal vertical rectangles side-by-side.
//!  - `DiagonalEdge`        : two equal tilted (45°) rectangles stacked.
//!  - `VerticalCenter`      : a center rectangle flanked by two side rectangles.
//!  - `HorizontalCenter`    : a center rectangle flanked by two top/bottom rectangles.

use crate::integral::{IntegralImage, RotatedIntegralImage};

/// One weighted sub-rectangle within a feature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    /// Top-left x in feature-local coordinates (0 ≤ x < feature width).
    pub x: u8,
    /// Top-left y.
    pub y: u8,
    /// Width in feature-local coordinates.
    pub w: u8,
    /// Height in feature-local coordinates.
    pub h: u8,
    /// Signed weight (sum of weights over a feature is 0).
    pub weight: f32,
}

impl Rect {
    pub const fn new(x: u8, y: u8, w: u8, h: u8, weight: f32) -> Self {
        Self { x, y, w, h, weight }
    }
}

/// Discriminator for the 5 canonical Haar feature families plus a
/// "custom-rects" variant that handles arbitrary OpenCV-style layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureKind {
    VerticalEdge,
    HorizontalEdge,
    DiagonalEdge,
    VerticalCenter,
    HorizontalCenter,
    /// Arbitrary rectangle layout (as stored in OpenCV's trained cascades).
    CustomRects,
}

/// A Haar feature: a feature kind + the layout of weighted rectangles inside it.
/// Feature-local coordinates are scaled by `step` pixels per unit when placed
/// in the integral image at position `(px, py)` and size `(w, h)`.
#[derive(Clone, Debug)]
pub struct HaarFeature {
    pub kind: FeatureKind,
    pub width: u8,  // feature-local width  (units)
    pub height: u8, // feature-local height (units)
    pub rects: Vec<Rect>,
}

impl HaarFeature {
    /// Evaluate the feature at integral-image position `(x, y)` covering window
    /// of pixel size `(win_w, win_h)`. Returns the raw response (caller decides
    /// how to threshold).
    ///
    /// For canonical features (VerticalEdge, etc.), `self.width` and `self.height`
    /// describe the feature's grid in feature-local coordinates (typically 1..N
    /// cells) and rect coordinates are mapped to window pixels via
    /// `(rx, ry) = (x + r.x * win_w / fw, y + r.y * win_h / fh)`.
    ///
    /// For `CustomRects` (OpenCV-style cascades), the rect coordinates are
    /// already in pixels relative to the top-left of the window, so no
    /// scaling is applied.
    ///
    /// Evaluate the feature response as a raw weighted sum:
///   `response = sum(weight_i * rect_sum_i)`
    ///
    /// Per OpenCV 4.x's `HaarEvaluator::OptFeature::calc` in
    /// `cascadedetect.hpp` the only normalization applied to the response is
    /// the per-window `varianceNormFactor` (the inverse sqrt of the variance
    /// of the inner normrect). There is **no** per-feature `normfactor`
    /// divided in at eval time — the older OpenCV code did this, and many
    /// third-party ports keep it, but the current OpenCV reference omits it.
    /// See https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/cascadedetect.hpp
    /// for the canonical reference.
    pub fn eval(&self, ii: &IntegralImage, ri: &RotatedIntegralImage,
                x: usize, y: usize, win_w: usize, win_h: usize,
                ii_w: usize, ii_h: usize) -> f32 {
        let mut total: f64 = 0.0;
        let is_custom = matches!(self.kind, FeatureKind::CustomRects);
        let fw = if is_custom { 1usize } else { self.width.max(1) as usize };
        let fh = if is_custom { 1usize } else { self.height.max(1) as usize };
        for r in &self.rects {
            let (rx, ry, rw, rh) = if is_custom {
                (x + r.x as usize, y + r.y as usize,
                 std::cmp::max(1, r.w as usize), std::cmp::max(1, r.h as usize))
            } else {
                let rx = x + r.x as usize * win_w / fw;
                let ry = y + r.y as usize * win_h / fh;
                let rw = std::cmp::max(1, r.w as usize * win_w / fw);
                let rh = std::cmp::max(1, r.h as usize * win_h / fh);
                (rx, ry, rw, rh)
            };
            let rx2 = (rx + rw).min(ii_w);
            let ry2 = (ry + rh).min(ii_h);
            let rx = rx.min(rx2);
            let ry = ry.min(ry2);
            let sum: i64 = match self.kind {
                FeatureKind::DiagonalEdge => ii.tilted_rect_sum(ri, rx, ry, rx2, ry2),
                _ => ii.rect_sum(rx, ry, rx2, ry2) as i64,
            };
            let contribution = (sum as f64) * (r.weight as f64);
            total += contribution;
        }
        total as f32
    }
}

/// Factory helpers — build the standard 5 features at a given feature size.
impl HaarFeature {
    /// Vertical edge: top half +1, bottom half -1.
    pub fn vertical_edge(fw: u8, fh: u8) -> Self {
        let half = fh / 2;
        Self {
            kind: FeatureKind::VerticalEdge,
            width: fw,
            height: fh,
            rects: vec![
                Rect::new(0, 0, fw, half, 1.0),
                Rect::new(0, half, fw, fh - half, -1.0),
            ],
        }
    }
    /// Horizontal edge: left half +1, right half -1.
    pub fn horizontal_edge(fw: u8, fh: u8) -> Self {
        let half = fw / 2;
        Self {
            kind: FeatureKind::HorizontalEdge,
            width: fw,
            height: fh,
            rects: vec![
                Rect::new(0, 0, half, fh, 1.0),
                Rect::new(half, 0, fw - half, fh, -1.0),
            ],
        }
    }
    /// Diagonal (tilted) edge: top-left +1, bottom-right -1.
    pub fn diagonal_edge(fw: u8, fh: u8) -> Self {
        Self {
            kind: FeatureKind::DiagonalEdge,
            width: fw,
            height: fh,
            rects: vec![
                Rect::new(0, 0, fw, fh / 2, 1.0),
                Rect::new(0, fh / 2, fw, fh - fh / 2, -1.0),
            ],
        }
    }
    /// Vertical center-surround: top +1, middle -2, bottom +1.
    pub fn vertical_center(fw: u8, fh: u8) -> Self {
        let third = fh / 3;
        Self {
            kind: FeatureKind::VerticalCenter,
            width: fw,
            height: fh,
            rects: vec![
                Rect::new(0, 0, fw, third, 1.0),
                Rect::new(0, third, fw, third, -2.0),
                Rect::new(0, 2 * third, fw, fh - 2 * third, 1.0),
            ],
        }
    }
    /// Horizontal center-surround: left +1, middle -2, right +1.
    pub fn horizontal_center(fw: u8, fh: u8) -> Self {
        let third = fw / 3;
        Self {
            kind: FeatureKind::HorizontalCenter,
            width: fw,
            height: fh,
            rects: vec![
                Rect::new(0, 0, third, fh, 1.0),
                Rect::new(third, 0, third, fh, -2.0),
                Rect::new(2 * third, 0, fw - 2 * third, fh, 1.0),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::GrayImage;

    #[test]
    fn vertical_edge_response() {
        // 2x2 image: top row = 0, bottom row = 255. Raw response =
        // (top sum) - (bottom sum) = 0 - 510 = -510. We then apply OpenCV's
        // normfactor = 1/(win_w * win_h) = 1/4 → -510/4 = -127.5.
        let mut img = GrayImage::new(2, 2);
        img[(0, 0)] = 0; img[(1, 0)] = 0;
        img[(0, 1)] = 255; img[(1, 1)] = 255;
        let ii = IntegralImage::from_gray(&img);
        let ri = RotatedIntegralImage::from_gray(&img);
        let feat = HaarFeature::vertical_edge(1, 2);
        let r = feat.eval(&ii, &ri, 0, 0, 2, 2, ii.width(), ii.height());
        assert_eq!(r, -127.5);
    }

    #[test]
    fn horizontal_edge_response() {
        // Same setup: raw -510, normalized by 1/4 → -127.5.
        let mut img = GrayImage::new(2, 2);
        img[(0, 0)] = 0; img[(1, 0)] = 255;
        img[(0, 1)] = 0; img[(1, 1)] = 255;
        let ii = IntegralImage::from_gray(&img);
        let ri = RotatedIntegralImage::from_gray(&img);
        let feat = HaarFeature::horizontal_edge(2, 1);
        let r = feat.eval(&ii, &ri, 0, 0, 2, 2, ii.width(), ii.height());
        assert_eq!(r, -127.5);
    }
}
