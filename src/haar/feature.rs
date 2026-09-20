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
    /// OpenCV `<tilted>` flag: every rect is a 45-degree rotated rectangle,
    /// so each rectangle sum is read from the rotated integral table rather
    /// than the upright one.
    pub tilted: bool,
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
    /// See <https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/cascadedetect.hpp>
    /// for the canonical reference.
    pub fn eval(
        &self,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        win_w: usize,
        win_h: usize,
        ii_w: usize,
        ii_h: usize,
    ) -> f32 {
        let mut total: f64 = 0.0;
        let is_custom = matches!(self.kind, FeatureKind::CustomRects);
        let is_tilted = self.tilted || matches!(self.kind, FeatureKind::DiagonalEdge);
        let fw = if is_custom {
            1usize
        } else {
            self.width.max(1) as usize
        };
        let fh = if is_custom {
            1usize
        } else {
            self.height.max(1) as usize
        };
        for r in &self.rects {
            let (rx, ry, rw, rh) = if is_custom {
                (
                    x + r.x as usize,
                    y + r.y as usize,
                    std::cmp::max(1, r.w as usize),
                    std::cmp::max(1, r.h as usize),
                )
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
            let sum: i64 = if is_tilted {
                ii.tilted_rect_sum(ri, rx, ry, rx2, ry2)
            } else {
                ii.rect_sum(rx, ry, rx2, ry2) as i64
            };
            let contribution = (sum as f64) * (r.weight as f64);
            total += contribution;
        }
        total as f32
    }

    /// Same as [`Self::eval`] but with the rectangle-sum clamping elided.
    ///
    /// # Safety contract (caller must uphold)
    /// For every rect `r` of this feature, `(rx + rw) <= ii_w` and
    /// `(ry + rh) <= ii_h` must hold **after** the same coordinate mapping
    /// `eval` performs (feature-local → window pixels → + (x, y)).
    ///
    /// This is the detector's window-scan configuration: `x + win_w <= ii_w`
    /// and `y + win_h <= ii_h` are guaranteed by the scan loop bounds, rects
    /// are subsets of the window (`rx ≥ x`, `ry ≥ y`, `rx + rw ≤ x + win_w`),
    /// and `ii_w/ii_h` equal the image dimensions, so every derived rect is
    /// inside the table. Under that contract the clamps in `eval` are
    /// identities and this function returns the exact same `f32` (each
    /// rect's sum and the f64 accumulation order are unchanged).
    ///
    /// Cascades whose rects overhang the window (possible with hand-edited
    /// `.rfcf` files) must keep using `eval`.
    ///
    /// ## Tilted features
    /// Window containment is NOT sufficient for tilted rects: their rotated
    /// lookup corners fan out left by the rect height and down by
    /// `width + height` (see [`RotatedIntegralImage::tilted_rect_sum`]).
    /// Near the image rims the corners overhang even though the upright rect
    /// fits; such rects fall back to the checked query, whose zero-border
    /// convention is the clipped result OpenCV computes from its padded
    /// rotated table. Interior rects keep the unchecked fast path, so a
    /// typical scan only pays for bounds checks on its rim windows.
    pub(crate) fn eval_inbounds(
        &self,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        win_w: usize,
        win_h: usize,
        ii_w: usize,
        ii_h: usize,
    ) -> f32 {
        debug_assert!(x + win_w <= ii_w && y + win_h <= ii_h);
        let mut total: f64 = 0.0;
        let is_custom = matches!(self.kind, FeatureKind::CustomRects);
        let is_tilted = self.tilted || matches!(self.kind, FeatureKind::DiagonalEdge);
        let fw = if is_custom {
            1usize
        } else {
            self.width.max(1) as usize
        };
        let fh = if is_custom {
            1usize
        } else {
            self.height.max(1) as usize
        };
        for r in &self.rects {
            let (rx, ry, rw, rh) = if is_custom {
                (
                    x + r.x as usize,
                    y + r.y as usize,
                    std::cmp::max(1, r.w as usize),
                    std::cmp::max(1, r.h as usize),
                )
            } else {
                let rx = x + r.x as usize * win_w / fw;
                let ry = y + r.y as usize * win_h / fh;
                let rw = std::cmp::max(1, r.w as usize * win_w / fw);
                let rh = std::cmp::max(1, r.h as usize * win_h / fh);
                (rx, ry, rw, rh)
            };
            // SAFETY (rect_sum_unchecked): rx < rx2 ≤ ii_w and ry < ry2 ≤ ii_h
            // follow from the documented contract of this method — rects map
            // inside the window, the window fits the image, and rw/rh ≥ 1.
            //
            // Dispatch the corner reads to the narrow (u32) specialised
            // path when the integral image is narrow — skips the
            // `IntegralTable` enum match that the generic variant emits,
            // and every realistic face-window input (640×480 up to 4K) is
            // narrow. The branch is one per `eval_inbounds` call, not per
            // rect.
            let sum: i64 = if is_tilted {
                // Tilted corners also have to clear the rotated-table rims:
                // p1 is `h` columns left of (rx, ry) and p3 is `rw + rh`
                // rows below it. Where they don't, the checked query clips
                // via its zero border — bit-identical on interior rects.
                let tilted_inbounds = rx >= rh && rx + rw <= ii_w && ry + rw + rh <= ii_h;
                if tilted_inbounds {
                    // SAFETY: all four tilted corners are inside the table.
                    ri.tilted_rect_sum_unchecked(rx, ry, rx + rw, ry + rh)
                } else {
                    ii.tilted_rect_sum(ri, rx, ry, rx + rw, ry + rh)
                }
            } else if ii.is_wide() {
                ii.rect_sum_unchecked(rx, ry, rx + rw, ry + rh) as i64
            } else {
                // SAFETY: caller guarantees the rect fits the window,
                // which fits the image; narrow contract follows from
                // `!is_wide()`.
                unsafe {
                    ii.rect_sum_unchecked_narrow(rx, ry, rx + rw, ry + rh) as i64
                }
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
            tilted: false,
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
            tilted: false,
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
            tilted: false,
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
            tilted: false,
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
            tilted: false,
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
        // (top sum) - (bottom sum) = 0 - 510 = -510.
        //
        // Per OpenCV 4.x's `HaarEvaluator::OptFeature::calc` in
        // cascadedetect.hpp the returned value is the raw weighted sum;
        // the per-window `varianceNormFactor` is applied at the call site,
        // and there is NO per-feature `1/(win_w * win_h)` division (older
        // OpenCV did that, modern does not). See commit f6e4849 for the
        // matching history on this file.
        let mut img = GrayImage::new(2, 2);
        img[(0, 0)] = 0;
        img[(1, 0)] = 0;
        img[(0, 1)] = 255;
        img[(1, 1)] = 255;
        let ii = IntegralImage::from_gray(&img);
        let ri = RotatedIntegralImage::from_gray(&img);
        let feat = HaarFeature::vertical_edge(1, 2);
        let r = feat.eval(&ii, &ri, 0, 0, 2, 2, ii.width(), ii.height());
        assert_eq!(r, -510.0);
    }

    #[test]
    fn horizontal_edge_response() {
        // Same setup: raw -510.
        let mut img = GrayImage::new(2, 2);
        img[(0, 0)] = 0;
        img[(1, 0)] = 255;
        img[(0, 1)] = 0;
        img[(1, 1)] = 255;
        let ii = IntegralImage::from_gray(&img);
        let ri = RotatedIntegralImage::from_gray(&img);
        let feat = HaarFeature::horizontal_edge(2, 1);
        let r = feat.eval(&ii, &ri, 0, 0, 2, 2, ii.width(), ii.height());
        assert_eq!(r, -510.0);
    }

    /// Sum of pixels in the tilted (45°) rectangle `(x1,y1,rw,rh)` using
    /// the OpenCV cone definition directly: cell R(X,Y) covers pixel
    /// (px,py) when `Y-1-py >= |X-1-px|`; the tilted query is
    /// R[p0]-R[p1]-R[p2]+R[p3] with the CV_TILTED_OFS corners.
    fn cone_rect_sum(img: &GrayImage, x1: usize, y1: usize, rw: usize, rh: usize) -> i64 {
        let (w, h) = (img.width() as isize, img.height() as isize);
        let cone = |cx: isize, cy: isize| -> i64 {
            if cx < 1 || cy < 1 || cx > w || cy > h {
                return 0;
            }
            let mut s = 0i64;
            for py in 0..h {
                for px in 0..w {
                    // (cy-1-py) >= |cx-1-px|  ⇔  cy-py > |cx-1-px|
                    if py < cy && cy - py > (cx - 1 - px).abs() {
                        s += img[(px as usize, py as usize)] as i64;
                    }
                }
            }
            s
        };
        let (x, y) = (x1 as isize, y1 as isize);
        let (rw, rh) = (rw as isize, rh as isize);
        cone(x, y) - cone(x - rh, y + rh) - cone(x + rw, y + rw) + cone(x + rw - rh, y + rw + rh)
    }

    #[test]
    fn tilted_custom_rects_eval_matches_open_cv_cone_sum() {
        // A converted OpenCV tilted feature: kind=CustomRects with the
        // `tilted` flag set must evaluate each rect through the rotated
        // integral table (CV_TILTED_OFS), not as an upright rect.
        let (w, h) = (48usize, 40usize);
        let mut img = GrayImage::new(w, h);
        let mut seed = 0x5EED_1234u32;
        for y in 0..h {
            for x in 0..w {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                img[(x, y)] = (seed >> 24) as u8;
            }
        }
        let ii = IntegralImage::from_gray(&img);
        let ri = RotatedIntegralImage::from_gray(&img);
        let feat = HaarFeature {
            kind: FeatureKind::CustomRects,
            width: 24,
            height: 24,
            tilted: true,
            rects: vec![Rect::new(4, 6, 8, 4, 1.0), Rect::new(4, 10, 8, 4, -1.0)],
        };
        let (ox, oy) = (2usize, 3usize);
        let got = feat.eval(&ii, &ri, ox, oy, 24, 24, w, h);
        let expected =
            cone_rect_sum(&img, ox + 4, oy + 6, 8, 4) - cone_rect_sum(&img, ox + 4, oy + 10, 8, 4);
        assert_eq!(got, expected as f32);
    }

    #[test]
    fn eval_inbounds_matches_eval_bit_for_bit() {
        // Deterministic pseudo-random image covering all feature families.
        let (w, h) = (48usize, 40usize);
        let mut img = GrayImage::new(w, h);
        let mut s = 0x0BAD_C0DEu32;
        for y in 0..h {
            for x in 0..w {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                img[(x, y)] = (s >> 24) as u8;
            }
        }
        let ii = IntegralImage::from_gray(&img);
        let ri = RotatedIntegralImage::from_gray(&img);
        let feats = vec![
            HaarFeature::vertical_edge(1, 2),
            HaarFeature::horizontal_edge(2, 1),
            HaarFeature::diagonal_edge(2, 2),
            HaarFeature::vertical_center(1, 3),
            HaarFeature::horizontal_center(3, 1),
        ];
        // Custom-rects feature (OpenCV style): rects in window pixels.
        let custom = HaarFeature {
            kind: FeatureKind::CustomRects,
            width: 24,
            height: 24,
            tilted: false,
            rects: vec![Rect::new(2, 2, 8, 8, 1.0), Rect::new(12, 4, 9, 10, -2.0)],
        };
        let mut all = feats;
        all.push(custom);
        // Windows fully inside the image — the detector's scan regime.
        for y in [0usize, 3, 9] {
            for x in [0usize, 5, 17] {
                for feat in &all {
                    let a = feat.eval(&ii, &ri, x, y, 24, 24, w, h);
                    let b = feat.eval_inbounds(&ii, &ri, x, y, 24, 24, w, h);
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "eval mismatch feat={:?} at ({x},{y})",
                        feat.kind
                    );
                }
            }
        }
    }
}
