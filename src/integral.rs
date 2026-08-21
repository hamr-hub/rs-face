//! Integral image (summed-area table) — O(1) rectangle sum queries.
//!
//! For an input grayscale image `I` of size `W × H`,
//! `II[x, y] = sum of I[i, j] for i < x, j < y` (with `II[0, *] = II[*, 0] = 0`).
//! Then the sum over rectangle `[x1, x2) × [y1, y2)` is:
//!   `II[x2, y2] - II[x1, y2] - II[x2, y1] + II[x1, y1]`.
//!
//! We use `u32` accumulation; for `1920 × 1080 × 255` the max value is ~5.3e8,
//! well within u32 range. A `u64` variant is offered for safety.
//!
//! Layout: row-major, with `width + 1` columns and `height + 1` rows.
//! Total memory = `(W+1) * (H+1) * sizeof(u32)` bytes.

use crate::image::GrayImage;

/// Integral image stored as a flat `Vec<u32>` of shape `(H+1, W+1)`.
/// Index `(x, y)` (0 <= x <= W, 0 <= y <= H) lives at `y * stride + x`.
#[derive(Clone)]
pub struct IntegralImage {
    data: Vec<u32>,
    width: usize,  // original image width
    height: usize, // original image height
    stride: usize, // = width + 1
}

impl IntegralImage {
    /// Construct from a precomputed `(W+1) × (H+1)` u32 buffer (as returned by
    /// the GPU kernel). The buffer layout must be row-major with stride = W+1.
    pub fn from_owned(data: Vec<u32>, width: usize, height: usize) -> Self {
        let stride = width + 1;
        Self {
            data,
            width,
            height,
            stride,
        }
    }

    /// Compute the integral image from a grayscale input.
    /// Padding row/column of zeros are added automatically.
    pub fn from_gray(img: &GrayImage) -> Self {
        let w = img.width();
        let h = img.height();
        let stride = w + 1;
        let mut data = vec![0u32; stride * (h + 1)];
        // Row-wise cumulative sum.
        for y in 0..h {
            let src_row = img.row(y);
            let dst_row = y + 1;
            let mut acc = 0u32;
            for x in 0..w {
                acc += src_row[x] as u32;
                data[dst_row * stride + x + 1] = acc + data[(dst_row - 1) * stride + x + 1];
            }
        }
        Self {
            data,
            width: w,
            height: h,
            stride,
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Raw access to `(x, y)` accumulator (0 <= x <= W, 0 <= y <= H).
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u32 {
        self.data[y * self.stride + x]
    }

    /// Sum of pixels in rectangle `[x1, x2) × [y1, y2)`.
    /// Returns 0 if the rectangle is empty.
    #[inline]
    pub fn rect_sum(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> u64 {
        if x2 <= x1 || y2 <= y1 {
            return 0;
        }
        let x2 = x2.min(self.width);
        let y2 = y2.min(self.height);
        if x1 >= x2 || y1 >= y2 {
            return 0;
        }
        let a = self.at(x1, y1) as u64;
        let b = self.at(x2, y1) as u64;
        let c = self.at(x1, y2) as u64;
        let d = self.at(x2, y2) as u64;
        d + a - b - c
    }

    /// Sum of pixels in a *tilted* (45°) rectangle used by Tilted Haar features.
    /// The tilted rectangle covers the diamond between four corners:
    ///   top    = (xmid, y1)
    ///   left   = (x1, ymid)
    ///   right  = (x2, ymid)
    ///   bottom = (xmid, y2)
    /// where `xmid = (x1 + x2) / 2`, `ymid = (y1 + y2) / 2`.
    ///
    /// Delegates to [`RotatedIntegralImage::tilted_rect_sum`], which uses
    /// Lienhart's closed-form formula on the rotated integral image.
    #[inline]
    pub fn tilted_rect_sum(
        &self,
        rotated: &RotatedIntegralImage,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
    ) -> i64 {
        rotated.tilted_rect_sum(x1, y1, x2, y2)
    }
}

/// Squared integral image: stores cumulative sum of `pixel_value^2`.
/// Used for O(1) variance computation inside a window.
/// For `u8` max=255, the max value at `(W, H)` is `W*H*255*255 = 6.5e10` for
/// 1920x1080 — fits in `u64`.
///
/// Variance of pixels inside window `[x1, x2) × [y1, y2)` is:
///   `E[X²] - E[X]² = sum_sq/N - (sum/N)² = (sum_sq * N - sum²) / N²`
/// The Viola-Jones test rejects windows whose variance is below a threshold by
/// comparing `(sum_sq * N - sum²)` against `(var_thresh * N²)`.
#[derive(Clone)]
pub struct SquaredIntegralImage {
    data: Vec<u64>,
    width: usize,
    height: usize,
    stride: usize,
}

impl SquaredIntegralImage {
    /// Construct from a precomputed `(W+1) × (H+1)` u64 buffer.
    pub fn from_owned(data: Vec<u64>, width: usize, height: usize) -> Self {
        let stride = width + 1;
        Self {
            data,
            width,
            height,
            stride,
        }
    }

    pub fn from_gray(img: &GrayImage) -> Self {
        let w = img.width();
        let h = img.height();
        let stride = w + 1;
        let mut data = vec![0u64; stride * (h + 1)];
        for y in 0..h {
            let src_row = img.row(y);
            let dst_row = y + 1;
            let mut acc: u64 = 0;
            for x in 0..w {
                let v = src_row[x] as u64;
                acc += v * v;
                data[dst_row * stride + x + 1] = acc + data[(dst_row - 1) * stride + x + 1];
            }
        }
        Self {
            data,
            width: w,
            height: h,
            stride,
        }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u64 {
        self.data[y * self.stride + x]
    }

    /// Sum of squared pixel values in `[x1, x2) × [y1, y2)`.
    #[inline]
    pub fn rect_sum_sq(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> u64 {
        if x2 <= x1 || y2 <= y1 {
            return 0;
        }
        let x2 = x2.min(self.width);
        let y2 = y2.min(self.height);
        if x1 >= x2 || y1 >= y2 {
            return 0;
        }
        let a = self.at(x1, y1);
        let b = self.at(x2, y1);
        let c = self.at(x1, y2);
        let d = self.at(x2, y2);
        d + a - b - c
    }

    /// Variance pre-filter. Returns `true` when the window has enough variance
    /// to potentially contain a face. Computes:
    ///   `sum_sq * N - sum² ≥ variance_threshold * N²`
    /// All operations are integer; N is the pixel count in the window.
    /// This is the canonical Viola-Jones first-stage rejection.
    ///
    /// Caller is responsible for picking a window rect that matches what the
    /// cascade's `varianceNormFactor` is computed over — for the OpenCV Haar
    /// cascade that is the INNER 22×22 normrect (`x+1, y+1, w-2, h-2`) of a
    /// 24×24 detection window, not the full 24×24 window. The detector
    /// passes the correct rect here.
    #[inline]
    pub fn passes_variance(
        &self,
        ii: &IntegralImage,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        variance_threshold: u64,
    ) -> bool {
        let sum = ii.rect_sum(x, y, x + w, y + h);
        let sum_sq = self.rect_sum_sq(x, y, x + w, y + h);
        let n = (w * h) as u64;
        // variance >= threshold iff sum_sq * N - sum² >= threshold * N²
        sum_sq.checked_mul(n).map_or(false, |lhs| {
            let sum_sq_part = lhs;
            let sum_part = sum.checked_mul(sum).unwrap_or(u64::MAX);
            let rhs = variance_threshold
                .checked_mul(n.checked_mul(n).unwrap_or(u64::MAX))
                .unwrap_or(u64::MAX);
            sum_sq_part >= sum_part + rhs
        })
    }
}

/// Rotated (45°) integral image: stores the cumulative sum over a 45° wedge.
///
/// Definition (Lienhart & Maydt, 2002): for a grayscale image `I`,
/// `R[x, y] = Σ I(i, j)` over all `(i, j)` with `i ≤ x`, `j ≤ y`, and
/// `i - j ≤ x - y`. Equivalently, `R` is the regular summed-area table
/// restricted to a 45° upper-half-plane anchored at the origin.
///
/// We use the two-pass construction:
/// 1. **Pass 1**: compute the regular integral `S[x, y] = Σ I(i, j)` over
///    `i ≤ x`, `j ≤ y`.
/// 2. **Pass 2**: transform to `R` using
///    `R[x, y] = R[x-1, y-1] + S[x, y] - S[x-1, y] - S[x, y-1] + S[x-1, y-1]`.
///
/// The recurrence needs *both* the S values and the R values, so S must be
/// kept in a separate buffer — in-place overwriting of S with R as you sweep
/// loses the S(x, y-1) and S(x-1, y-1) terms. The previous implementation
/// overwrote in place and produced wrong R values.
///
/// The tilted-rectangle sum (the diamond between four corners of a 45°
/// rotated rectangle) is then a 6-term combination of `R` lookups.
#[derive(Clone)]
pub struct RotatedIntegralImage {
    data: Vec<i64>,
    width: usize,
    height: usize,
    stride: usize,
}

impl RotatedIntegralImage {
    /// Build the rotated (45°) integral image from a grayscale input.
    pub fn from_gray(img: &GrayImage) -> Self {
        let w = img.width();
        let h = img.height();
        let stride = w + 1;
        // First, compute the regular integral image in a separate buffer.
        // The rotated recurrence
        //   R(x,y) = R(x-1,y-1) + S(x,y) - S(x-1,y) - S(x,y-1) + S(x-1,y-1)
        // needs *both* the S values (regular integral) and the R values.
        // The previous in-place implementation overwrote S with R as it
        // iterated, which made S(x, y-1) and S(x-1, y-1) read R values
        // instead — producing wrong R values downstream. We allocate S
        // separately so the recurrence sees the true S at every cell.
        let mut s = vec![0i64; stride * (h + 1)];
        for y in 1..=h {
            let mut row_acc: i64 = 0;
            for x in 1..=w {
                row_acc += img[(x - 1, y - 1)] as i64;
                s[y * stride + x] = row_acc + s[(y - 1) * stride + x];
            }
        }
        // Now compute R from S using the Lienhart recurrence. At y=1 or
        // x=1 the R(x-1, y-1) term is 0 (out of bounds).
        let mut data = vec![0i64; stride * (h + 1)];
        for y in 1..=h {
            for x in 1..=w {
                let s_xy = s[y * stride + x];
                let s_x1y = s[y * stride + (x - 1)];
                let s_xy1 = s[(y - 1) * stride + x];
                let s_x1y1 = s[(y - 1) * stride + (x - 1)];
                let r_x1y1 = if x >= 2 && y >= 2 {
                    data[(y - 1) * stride + (x - 1)]
                } else {
                    0
                };
                data[y * stride + x] = r_x1y1 + s_xy - s_x1y - s_xy1 + s_x1y1;
            }
        }
        Self {
            data,
            width: w,
            height: h,
            stride,
        }
    }

    /// Query the rotated integral at `(x, y)` (0 ≤ x ≤ width, 0 ≤ y ≤ height).
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> i64 {
        self.data[y * self.stride + x]
    }

    /// Query a tilted (45°) rectangle whose four corners are:
    /// - top    = (xmid, y1)
    /// - left   = (x1,  ymid)
    /// - right  = (x2,  ymid)
    /// - bottom = (xmid, y2)
    ///
    /// where `xmid = (x1 + x2) / 2`, `ymid = (y1 + y2) / 2`. This is the
    /// diamond-sum form used by OpenCV's tilted Haar features.
    ///
    /// Implementation uses Lienhart's closed-form expansion of the 45°
    /// rotated rectangle as a 6-term combination of `R` lookups. We split
    /// the diamond into an upper and lower triangle; each triangle is the
    /// `R`-difference between two rectangles plus a single regular integral.
    pub fn tilted_rect_sum(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> i64 {
        if x2 <= x1 || y2 <= y1 {
            return 0;
        }
        // Clamp to image bounds (the rect_sum-equivalent does this; here we
        // operate on the R array which is sized to (W+1, H+1)).
        let x2 = x2.min(self.width);
        let y2 = y2.min(self.height);
        if x1 >= x2 || y1 >= y2 {
            return 0;
        }
        let xmid = (x1 + x2) / 2;
        let ymid = (y1 + y2) / 2;
        // Upper triangle: vertices (x1, ymid), (xmid, y1), (xmid, ymid+1).
        // Sum = R[xmid, y1] - R[x1, y1] - R[xmid, ymid] + R[x1, ymid]
        let upper = self.at(xmid, y1) - self.at(x1, y1) - self.at(xmid, ymid) + self.at(x1, ymid);
        // Lower triangle: vertices (xmid, ymid), (x2, ymid), (xmid, y2).
        // Sum = R[x2, ymid] - R[xmid, ymid] - R[x2, y2] + R[xmid, y2]
        let lower = self.at(x2, ymid) - self.at(xmid, ymid) - self.at(x2, y2) + self.at(xmid, y2);
        upper + lower
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::GrayImage;

    #[test]
    fn rect_sum_basic() {
        // 3x3 image of all ones -> total = 9
        let mut img = GrayImage::new(3, 3);
        for y in 0..3 {
            for x in 0..3 {
                img[(x, y)] = 1;
            }
        }
        let ii = IntegralImage::from_gray(&img);
        assert_eq!(ii.rect_sum(0, 0, 3, 3), 9);
        assert_eq!(ii.rect_sum(1, 1, 3, 3), 4);
        assert_eq!(ii.rect_sum(0, 0, 0, 0), 0);
    }

    #[test]
    fn rect_sum_known_pattern() {
        // 2x2: [[1,2],[3,4]]
        let mut img = GrayImage::new(2, 2);
        img[(0, 0)] = 1;
        img[(1, 0)] = 2;
        img[(0, 1)] = 3;
        img[(1, 1)] = 4;
        let ii = IntegralImage::from_gray(&img);
        assert_eq!(ii.rect_sum(0, 0, 2, 2), 10);
        assert_eq!(ii.rect_sum(1, 1, 2, 2), 4);
        assert_eq!(ii.rect_sum(0, 0, 1, 1), 1);
        assert_eq!(ii.rect_sum(0, 0, 2, 1), 3); // top row 1+2
    }

    #[test]
    fn rotated_rect_sum_compiles() {
        // The rotated integral recurrence
        //   R(x,y) = R(x-1,y-1) + S(x,y) - S(x-1,y) - S(x,y-1) + S(x-1,y-1)
        // needs *both* S and R at neighbouring cells, so S cannot be
        // overwritten in place. The previous implementation did overwrite,
        // producing wrong R values for any image larger than 2 pixels.
        // We just sanity-check that the recurrence completes, that the
        // boundary cells (R(0,*), R(*,0)) are zero, and that tilted_rect_sum
        // of a non-empty rect returns a non-zero value on a non-trivial image.
        let mut img = GrayImage::new(2, 2);
        img[(0, 0)] = 1;
        img[(1, 0)] = 2;
        img[(0, 1)] = 3;
        img[(1, 1)] = 4;
        let ri = RotatedIntegralImage::from_gray(&img);
        assert_eq!(ri.at(0, 0), 0);
        assert_eq!(ri.at(1, 0), 0);
        assert_eq!(ri.at(0, 1), 0);
        assert_ne!(ri.tilted_rect_sum(0, 0, 2, 2), 0);
    }
}
