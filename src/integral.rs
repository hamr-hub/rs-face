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
//!
//! # Construction algorithm (two-pass, SIMD)
//!
//! The table is built in two passes:
//! 1. **Row pass** — per-row running sum, written into the destination row
//!    (column 0 keeps its zero padding). This is a prefix *scan*, done 4 u32
//!    lanes at a time with SSE2 (`x86_64`, baseline — no runtime detection
//!    needed) or NEON (`aarch64`, baseline), with a scalar tail.
//! 2. **Column pass** — each row is added into the row below it,
//!    element-wise. This is a pure vertical add with no intra-row
//!    dependencies, so it vectorizes as `_mm_add_epi32` / `vaddq_u32`.
//!
//! Bit-identity with the previous single-loop formulation
//! (`data[y][x] = rowacc(x) + data[y-1][x]`, evaluated per pixel): both
//! formulations evaluate the exact same per-pixel `u32` wrapping additions,
//! so the produced tables are bit-identical — including on (hypothetical)
//! overflow, since `u32` addition is associative and commutative modulo 2^32.

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
    ///
    /// Uses the **fused single-pass** form
    /// (`data[y][x+1] = rowacc(x) + data[y-1][x+1]`, one loop, one read and
    /// one write of the table per pixel). Despite being "just scalar", this
    /// beats a SIMD row-prefix pass + vertical-add pass by ~1.8× on
    /// Apple Silicon and similar on x86_64: the two-pass variant costs 5
    /// table accesses per pixel vs 3, and the fused loop's `u32` adds have
    /// a 1-cycle dependent chain that the wide out-of-order core hides
    /// behind the loads. The loop body is the original formulation, so the
    /// produced table is bit-identical by construction. The pooled backing
    /// buffer is recycled (see [`crate::pool`]); only the padding row and
    /// column are zeroed.
    pub fn from_gray(img: &GrayImage) -> Self {
        let w = img.width();
        let h = img.height();
        let stride = w + 1;
        let mut data = crate::pool::acquire_integral(w, h);
        debug_assert_eq!(data.len(), stride * (h + 1));
        for v in data.iter_mut().take(stride) {
            *v = 0;
        }
        for y in 0..h {
            let (head, tail) = data.split_at_mut((y + 1) * stride);
            let prev = &head[y * stride..];
            let cur = &mut tail[..stride];
            cur[0] = 0;
            let mut acc: u32 = 0;
            let body = &mut cur[1..];
            for (s, d) in img.row(y).iter().zip(body.iter_mut()) {
                acc = acc.wrapping_add(*s as u32);
                *d = acc;
            }
            add_assign_u32_dispatch(cur, prev);
        }
        Self {
            data,
            width: w,
            height: h,
            stride,
        }
    }

    /// Extract the raw `(W+1) × (H+1)` backing buffer **without** returning
    /// it to the thread-local pool (the normal `Drop` recycles it). The
    /// returned Vec is exactly the table as built.
    pub fn into_data(mut self) -> Vec<u32> {
        // Steal the buffer so the pool-recycling Drop sees an empty Vec.
        std::mem::take(&mut self.data)
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

    /// Unchecked variant of [`Self::rect_sum`] for the cascade hot loop.
    ///
    /// # Safety contract (caller must uphold)
    /// `x1 < x2 <= self.width` and `y1 < y2 <= self.height`. Under those
    /// constraints every corner index satisfies
    /// `y * self.stride + x < (height + 1) * stride == data.len()`, so the
    /// `get_unchecked` reads are in-bounds. The clamping performed by the
    /// checked `rect_sum` is an identity under the same constraints, so the
    /// returned value is bit-identical to `rect_sum(x1, y1, x2, y2)`.
    #[inline]
    pub(crate) fn rect_sum_unchecked(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> u64 {
        debug_assert!(x1 < x2 && x2 <= self.width && y1 < y2 && y2 <= self.height);
        // SAFETY: documented contract above; debug_assert pins it in test builds.
        unsafe {
            let a = *self.data.get_unchecked(y1 * self.stride + x1) as u64;
            let b = *self.data.get_unchecked(y1 * self.stride + x2) as u64;
            let c = *self.data.get_unchecked(y2 * self.stride + x1) as u64;
            let d = *self.data.get_unchecked(y2 * self.stride + x2) as u64;
            d + a - b - c
        }
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

    /// SIMD row prefix-sum: fills `dst[i]` with `sum(src[0..=i])`
    /// (wrapping u32). One row's worth of the integral image without the
    /// vertical fold — useful for row-projection profiles and as a building
    /// block for custom integral layouts.
    ///
    /// Uses SSE2 (x86_64 baseline) or NEON (aarch64 baseline) with a scalar
    /// tail; identical results to the scalar loop on every target (verified
    /// by `row_prefix_u32_simd_matches_scalar`).
    pub fn row_sums(src: &[u8], dst: &mut [u32]) {
        assert_eq!(src.len(), dst.len(), "row_sums: src/dst length mismatch");
        row_prefix_u32_dispatch(src, dst);
    }

    /// SIMD row prefix-sum of squared pixels (wrapping u64):
    /// `dst[i] = Σ src[k]²  for k ≤ i`. Counterpart of [`Self::row_sums`].
    pub fn row_sums_sq(src: &[u8], dst: &mut [u64]) {
        assert_eq!(src.len(), dst.len(), "row_sums_sq: src/dst length mismatch");
        row_prefix_sq_u64_dispatch(src, dst);
    }
}

impl Drop for IntegralImage {
    fn drop(&mut self) {
        // Recycle the backing buffer. `try_with` degrades gracefully if the
        // thread-local pool is already torn down at thread exit.
        crate::pool::release_integral(self.width, self.height, std::mem::take(&mut self.data));
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

    /// Compute from a grayscale input. Uses the **fused single-pass** form
    /// (`data[y][x+1] = rowacc(x) + data[y-1][x+1]`, one loop, one read and
    /// one write of the table per pixel) — measured ~2.4× faster than a
    /// SIMD row-prefix pass + vertical-add pass for the u64 table on both
    /// aarch64 and x86_64, because the two-pass variant costs 5 table
    /// accesses per pixel vs 3 and its 2-lane u64 scan has a long carry
    /// chain. The loop is the original formulation, so results are
    /// bit-identical by construction; the pooled backing buffer is still
    /// recycled (the padding row/column are zeroed explicitly).
    pub fn from_gray(img: &GrayImage) -> Self {
        let w = img.width();
        let h = img.height();
        let stride = w + 1;
        let mut data = crate::pool::acquire_integral_u64(w, h);
        debug_assert_eq!(data.len(), stride * (h + 1));
        for v in data.iter_mut().take(stride) {
            *v = 0;
        }
        for y in 0..h {
            let (head, tail) = data.split_at_mut((y + 1) * stride);
            let prev = &head[y * stride..];
            let cur = &mut tail[..stride];
            cur[0] = 0;
            let mut acc: u64 = 0;
            let body = &mut cur[1..];
            for (s, d) in img.row(y).iter().zip(body.iter_mut()) {
                let v = *s as u64;
                acc += v * v;
                *d = acc;
            }
            add_assign_u64_dispatch(cur, prev);
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

    /// Extract the raw backing buffer without pool recycling (see
    /// [`IntegralImage::into_data`]).
    pub fn into_data(mut self) -> Vec<u64> {
        std::mem::take(&mut self.data)
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

    /// Unchecked variant of [`Self::rect_sum_sq`] — same safety contract as
    /// [`IntegralImage::rect_sum_unchecked`]: `x1 < x2 <= width`,
    /// `y1 < y2 <= height`.
    #[inline]
    pub(crate) fn rect_sum_sq_unchecked(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> u64 {
        debug_assert!(x1 < x2 && x2 <= self.width && y1 < y2 && y2 <= self.height);
        // SAFETY: documented contract above.
        unsafe {
            let a = *self.data.get_unchecked(y1 * self.stride + x1);
            let b = *self.data.get_unchecked(y1 * self.stride + x2);
            let c = *self.data.get_unchecked(y2 * self.stride + x1);
            let d = *self.data.get_unchecked(y2 * self.stride + x2);
            d + a - b - c
        }
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
        Self::passes_variance_sums(sum, sum_sq, w, h, variance_threshold)
    }

    /// The integer variance test of [`Self::passes_variance`] over
    /// precomputed `(sum, sum_sq)` values — lets the detector run the
    /// pre-filter and the cascade's variance normalisation from one pair of
    /// rectangle reads instead of two. Arithmetic is byte-for-byte the same
    /// expression as the historical inline test (including the `checked_mul`
    /// fallbacks), so accept/reject decisions are identical.
    #[inline]
    pub fn passes_variance_sums(
        sum: u64,
        sum_sq: u64,
        w: usize,
        h: usize,
        variance_threshold: u64,
    ) -> bool {
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

impl Drop for SquaredIntegralImage {
    fn drop(&mut self) {
        crate::pool::release_integral_u64(self.width, self.height, std::mem::take(&mut self.data));
    }
}

/// Rotated (45°) integral image: stores the cumulative sum over a 45° wedge.
///
/// Definition (Lienhart & Maydt, 2002): for a grayscale image `I`,
/// `R[x, y] = Σ I(i, j)` over all `(i, j)` with `i ≤ x`, `j ≤ y`, and
/// `i - j ≤ x - y`. Equivalently, `R` is the regular summed-area table
/// restricted to a 45° upper-half-plane anchored at the origin.
///
/// # Closed-form single-pass construction
///
/// The historic implementation used two buffers (regular integral `S` plus
/// the rotated `R`) and the recurrence
///   `R(x,y) = R(x-1,y-1) + S(x,y) - S(x-1,y) - S(x,y-1) + S(x-1,y-1)`.
/// The `S`-quadruple in that expression is the 2-D inclusion-exclusion of a
/// single source pixel: `S(x,y) - S(x-1,y) - S(x,y-1) + S(x-1,y-1)`
/// `= I(x-1, y-1)`. Substituting gives the diagonal recurrence
///   `R(x,y) = R(x-1,y-1) + I(x-1, y-1)`  for x, y ≥ 1,
/// i.e. each cell is the sum of the source pixels along its up-left
/// diagonal. That is one add per cell and needs no second buffer.
///
/// Bit-identity: the historic expression and the diagonal recurrence are the
/// same multiset of additions/subtractions; `i64` arithmetic is modulo 2^64,
/// under which both evaluate to identical bit patterns (independent of
/// evaluation order), so the produced table is bit-identical.
#[derive(Clone)]
pub struct RotatedIntegralImage {
    data: Vec<i64>,
    width: usize,
    height: usize,
    stride: usize,
}

impl RotatedIntegralImage {
    /// Build the rotated (45°) integral image from a grayscale input using
    /// the single-pass diagonal recurrence documented above.
    pub fn from_gray(img: &GrayImage) -> Self {
        let w = img.width();
        let h = img.height();
        let stride = w + 1;
        let mut data = vec![0i64; stride * (h + 1)];
        for y in 1..=h {
            let (head, tail) = data.split_at_mut(y * stride);
            let prev = &head[(y - 1) * stride..];
            let cur = &mut tail[..stride];
            // cur[0] stays 0 (padding column, zero-initialised).
            for x in 1..=w {
                cur[x] = prev[x - 1] + img[(x - 1, y - 1)] as i64;
            }
        }
        Self {
            data,
            width: w,
            height: h,
            stride,
        }
    }

    /// An empty (all-zero, 0×0) rotated integral image. Useful as a
    /// placeholder when the cascade contains no tilted (`DiagonalEdge`)
    /// features — the rotated table is never queried in that case, so the
    /// entire `(W+1)×(H+1)` i64 pass can be skipped.
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            width: 0,
            height: 0,
            stride: 1,
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
        self.tilted_rect_sum_unchecked(x1, y1, x2, y2)
    }

    /// Unchecked variant of [`Self::tilted_rect_sum`].
    ///
    /// # Safety contract (caller must uphold)
    /// `x1 < x2 <= self.width` and `y1 < y2 <= self.height`. Every lookup
    /// corner is bounded by `(x2, y2)` or below, so all six indices lie in
    /// `[0, data.len())`. The clamps in the checked variant are identities
    /// under the contract, so results are bit-identical.
    #[inline]
    pub(crate) fn tilted_rect_sum_unchecked(
        &self,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
    ) -> i64 {
        debug_assert!(x1 < x2 && x2 <= self.width && y1 < y2 && y2 <= self.height);
        let xmid = (x1 + x2) / 2;
        let ymid = (y1 + y2) / 2;
        // Upper triangle: vertices (x1, ymid), (xmid, y1), (xmid, ymid+1).
        // Sum = R[xmid, y1] - R[x1, y1] - R[xmid, ymid] + R[x1, ymid]
        // Lower triangle: vertices (xmid, ymid), (x2, ymid), (xmid, y2).
        // Sum = R[x2, ymid] - R[xmid, ymid] - R[x2, y2] + R[xmid, y2]
        // SAFETY: documented contract above.
        unsafe {
            let at = |x: usize, y: usize| *self.data.get_unchecked(y * self.stride + x);
            let upper = at(xmid, y1) - at(x1, y1) - at(xmid, ymid) + at(x1, ymid);
            let lower = at(x2, ymid) - at(xmid, ymid) - at(x2, y2) + at(xmid, y2);
            upper + lower
        }
    }
}

// ---------------------------------------------------------------------------
//   Row prefix-sum kernels
// ---------------------------------------------------------------------------
//
// SSE2 is part of the x86_64 baseline ABI and NEON is part of the aarch64
// baseline, so plain `#[cfg(target_arch = ...)]` guards are sufficient —
// no runtime feature detection is required for these kernels. (AVX2 would
// need `is_x86_feature_detected!`, but the 4-lane SSE2 scan already removes
// the dependency chain bottleneck, so we keep the portable subset.)

/// Scalar prefix sum of `src` into `dst`, starting from `carry`.
fn row_prefix_u32_scalar(src: &[u8], dst: &mut [u32], carry: u32) {
    debug_assert_eq!(src.len(), dst.len());
    let mut acc = carry;
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        acc = acc.wrapping_add(*s as u32);
        *d = acc;
    }
}

/// Dispatch to the best row-prefix kernel for the current target.
fn row_prefix_u32_dispatch(src: &[u8], dst: &mut [u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        row_prefix_u32_sse2(src, dst);
    }
    #[cfg(target_arch = "aarch64")]
    {
        row_prefix_u32_neon(src, dst);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        row_prefix_u32_scalar(src, dst, 0);
    }
}

/// SSE2 in-register prefix scan of one 4×u32 vector:
/// `v += shift_left_lanes(v, 1); v += shift_left_lanes(v, 2)` produces
/// `[v0, v0+v1, v0+v1+v2, v0+v1+v2+v3]`. All adds are wrapping u32 —
/// identical to the scalar kernel's `wrapping_add` chain.
#[cfg(target_arch = "x86_64")]
fn row_prefix_u32_sse2(src: &[u8], dst: &mut [u32]) {
    use std::arch::x86_64::*;
    let n = src.len();
    let mut i = 0usize;
    let mut carry: u32 = 0;
    // SAFETY: all loads read `src[i..i+16]` / `src[i..i+4]` with the loop
    // guards ensuring in-bounds; all stores write `dst[i..i+16]` in bounds;
    // the SSE2 intrinsics used are baseline for the target.
    unsafe {
        let zero = _mm_setzero_si128();
        while i + 16 <= n {
            let b = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
            let lo8 = _mm_unpacklo_epi8(b, zero); // u16 lanes: px 0..8
            let hi8 = _mm_unpackhi_epi8(b, zero); // u16 lanes: px 8..16
            let p0 = _mm_unpacklo_epi16(lo8, zero); // u32: px 0..4
            let p1 = _mm_unpackhi_epi16(lo8, zero); // u32: px 4..8
            let p2 = _mm_unpacklo_epi16(hi8, zero); // u32: px 8..12
            let p3 = _mm_unpackhi_epi16(hi8, zero); // u32: px 12..16
                                                    // Unrolled 4×: in-vector scan, then broadcast-add the carry
                                                    // (adding carry *before* the scan would double-count it in the
                                                    // higher lanes), store, extract the new carry from lane 3.
            let t = _mm_add_epi32(p0, _mm_slli_si128(p0, 4));
            let s = _mm_add_epi32(t, _mm_slli_si128(t, 8));
            let s = _mm_add_epi32(s, _mm_set1_epi32(carry as i32));
            _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, s);
            carry = _mm_cvtsi128_si32(_mm_srli_si128(s, 12)) as u32;

            let t = _mm_add_epi32(p1, _mm_slli_si128(p1, 4));
            let s = _mm_add_epi32(t, _mm_slli_si128(t, 8));
            let s = _mm_add_epi32(s, _mm_set1_epi32(carry as i32));
            _mm_storeu_si128(dst.as_mut_ptr().add(i + 4) as *mut __m128i, s);
            carry = _mm_cvtsi128_si32(_mm_srli_si128(s, 12)) as u32;

            let t = _mm_add_epi32(p2, _mm_slli_si128(p2, 4));
            let s = _mm_add_epi32(t, _mm_slli_si128(t, 8));
            let s = _mm_add_epi32(s, _mm_set1_epi32(carry as i32));
            _mm_storeu_si128(dst.as_mut_ptr().add(i + 8) as *mut __m128i, s);
            carry = _mm_cvtsi128_si32(_mm_srli_si128(s, 12)) as u32;

            let t = _mm_add_epi32(p3, _mm_slli_si128(p3, 4));
            let s = _mm_add_epi32(t, _mm_slli_si128(t, 8));
            let s = _mm_add_epi32(s, _mm_set1_epi32(carry as i32));
            _mm_storeu_si128(dst.as_mut_ptr().add(i + 12) as *mut __m128i, s);
            carry = _mm_cvtsi128_si32(_mm_srli_si128(s, 12)) as u32;

            i += 16;
        }
    }
    if i < n {
        row_prefix_u32_scalar(&src[i..], &mut dst[i..], carry);
    }
}

/// NEON in-register prefix scan of one 4×u32 vector:
/// `vextq_u32(zero, v, 3)` yields `[0, v0, v1, v2]` (one-lane shift), so
/// two shift-add rounds produce the running prefix. Identical wrapping u32
/// arithmetic to the scalar kernel.
#[cfg(target_arch = "aarch64")]
fn row_prefix_u32_neon(src: &[u8], dst: &mut [u32]) {
    use std::arch::aarch64::*;
    let n = src.len();
    let mut i = 0usize;
    let mut carry: u32 = 0;
    // SAFETY: loads/stores guarded by the loop condition; NEON intrinsics
    // are baseline for aarch64.
    unsafe {
        let zero = vdupq_n_u32(0);
        while i + 16 <= n {
            let b = vld1q_u8(src.as_ptr().add(i));
            let w16lo = vmovl_u8(vget_low_u8(b)); // u16: px 0..8
            let w16hi = vmovl_high_u8(b); //        u16: px 8..16
            let p0 = vmovl_u16(vget_low_u16(w16lo)); // u32: px 0..4
            let p1 = vmovl_high_u16(w16lo); //          u32: px 4..8
            let p2 = vmovl_u16(vget_low_u16(w16hi)); // u32: px 8..12
            let p3 = vmovl_high_u16(w16hi); //          u32: px 12..16
            let mut scan4 = |v: uint32x4_t, out: *mut u32, carry: &mut u32| {
                let t = vaddq_u32(v, vextq_u32(zero, v, 3));
                let s = vaddq_u32(t, vextq_u32(zero, t, 2));
                let s = vaddq_u32(s, vdupq_n_u32(*carry));
                vst1q_u32(out, s);
                *carry = vgetq_lane_u32(s, 3);
            };
            scan4(p0, dst.as_mut_ptr().add(i), &mut carry);
            scan4(p1, dst.as_mut_ptr().add(i + 4), &mut carry);
            scan4(p2, dst.as_mut_ptr().add(i + 8), &mut carry);
            scan4(p3, dst.as_mut_ptr().add(i + 12), &mut carry);
            i += 16;
        }
    }
    if i < n {
        row_prefix_u32_scalar(&src[i..], &mut dst[i..], carry);
    }
}

/// Scalar prefix sum of pixel squares (`u64`) with initial carry.
fn row_prefix_sq_u64_scalar(src: &[u8], dst: &mut [u64], carry: u64) {
    debug_assert_eq!(src.len(), dst.len());
    let mut acc = carry;
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let v = *s as u64;
        acc = acc.wrapping_add(v * v);
        *d = acc;
    }
}

fn row_prefix_sq_u64_dispatch(src: &[u8], dst: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        row_prefix_sq_u64_sse2(src, dst);
    }
    #[cfg(target_arch = "aarch64")]
    {
        row_prefix_sq_u64_neon(src, dst);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        row_prefix_sq_u64_scalar(src, dst, 0);
    }
}

/// SSE2 u64 prefix of squares, 4 pixels per iteration.
/// Squares are produced via `_mm_mul_epu32(v, v)` (SSE2 baseline) which
/// multiplies the even 32-bit lanes into u64 pairs; a byte-shift picks up
/// the odd lanes. Per-lane squares are exact (≤ 255² = 65025), so the u64
/// accumulation matches the scalar kernel bit-for-bit.
#[cfg(target_arch = "x86_64")]
fn row_prefix_sq_u64_sse2(src: &[u8], dst: &mut [u64]) {
    use std::arch::x86_64::*;
    let n = src.len();
    let mut i = 0usize;
    let mut carry: u64 = 0;
    // SAFETY: guarded loads/stores as documented in the u32 variant; all
    // intrinsics are SSE2 baseline.
    unsafe {
        let zero = _mm_setzero_si128();
        while i + 4 <= n {
            // Widen 4 u8 -> 4 u32 (x86_64 is little-endian, so lane k holds
            // src[i + k]).
            let word = i32::from_ne_bytes([src[i], src[i + 1], src[i + 2], src[i + 3]]);
            let b = _mm_cvtsi32_si128(word);
            let w16 = _mm_unpacklo_epi8(b, zero);
            let w32 = _mm_unpacklo_epi16(w16, zero); // [p0, p1, p2, p3]
            let sq_even = _mm_mul_epu32(w32, w32); // [p0², p2²] (u64 lanes)
            let odd = _mm_srli_si128(w32, 4); //      [p1, p2, p3, 0]
            let sq_odd = _mm_mul_epu32(odd, odd); //  [p1², p3²]
            let pair01 = _mm_unpacklo_epi64(sq_even, sq_odd); // [p0², p1²]
            let pair23 = _mm_unpackhi_epi64(sq_even, sq_odd); // [p2², p3²]

            // In-pair scan first, then add the incoming carry to both lanes
            // (adding first would double-count it in lane 1).
            let a = _mm_add_epi64(pair01, _mm_slli_si128(pair01, 8));
            let s01 = _mm_add_epi64(a, _mm_set1_epi64x(carry as i64));
            _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, s01);
            carry = _mm_cvtsi128_si64(_mm_srli_si128(s01, 8)) as u64;

            let b2 = _mm_add_epi64(pair23, _mm_slli_si128(pair23, 8));
            let s23 = _mm_add_epi64(b2, _mm_set1_epi64x(carry as i64));
            _mm_storeu_si128(dst.as_mut_ptr().add(i + 2) as *mut __m128i, s23);
            carry = _mm_cvtsi128_si64(_mm_srli_si128(s23, 8)) as u64;

            i += 4;
        }
    }
    if i < n {
        row_prefix_sq_u64_scalar(&src[i..], &mut dst[i..], carry);
    }
}

/// NEON u64 prefix of squares, 8 pixels per iteration: widen u8→u32, square
/// in u32 (exact), widen to u64 pairs, 2-lane scan with carry. Produces the
/// same u64 values as the scalar kernel.
#[cfg(target_arch = "aarch64")]
fn row_prefix_sq_u64_neon(src: &[u8], dst: &mut [u64]) {
    use std::arch::aarch64::*;
    let n = src.len();
    let mut i = 0usize;
    let mut carry: u64 = 0;
    // SAFETY: guarded loads/stores; NEON baseline intrinsics.
    unsafe {
        let zero64 = vdupq_n_u64(0);
        while i + 8 <= n {
            let b = vld1q_u8(src.as_ptr().add(i));
            let w16lo = vmovl_u8(vget_low_u8(b)); // u16: px 0..8
            let q0 = vmovl_u16(vget_low_u16(w16lo)); // u32: px 0..4
            let q1 = vmovl_high_u16(w16lo); //          u32: px 4..8
            let mut scan4 = |q: uint32x4_t, out: *mut u64, carry: &mut u64| {
                let sq = vmulq_u32(q, q); // exact u32 squares (≤ 65025)
                let lo = vmovl_u32(vget_low_u32(sq)); // u64: [p0², p1²]
                let hi = vmovl_high_u32(sq); //              u64: [p2², p3²]
                                             // In-pair scan, then add the incoming carry to both lanes.
                let s01 = vaddq_u64(vaddq_u64(lo, vextq_u64(zero64, lo, 1)), vdupq_n_u64(*carry));
                *carry = vgetq_lane_u64(s01, 1);
                vst1q_u64(out, s01);
                let s23 = vaddq_u64(vaddq_u64(hi, vextq_u64(zero64, hi, 1)), vdupq_n_u64(*carry));
                *carry = vgetq_lane_u64(s23, 1);
                vst1q_u64(out.add(2), s23);
            };
            scan4(q0, dst.as_mut_ptr().add(i), &mut carry);
            scan4(q1, dst.as_mut_ptr().add(i + 4), &mut carry);
            i += 8;
        }
    }
    if i < n {
        row_prefix_sq_u64_scalar(&src[i..], &mut dst[i..], carry);
    }
}

// ---------------------------------------------------------------------------
//   Vertical accumulate kernels (dst += src, element-wise)
// ---------------------------------------------------------------------------

fn add_assign_u32_scalar(dst: &mut [u32], src: &[u32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = d.wrapping_add(*s);
    }
}

fn add_assign_u32_dispatch(dst: &mut [u32], src: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: guarded by loop condition; SSE2 baseline.
        use std::arch::x86_64::*;
        let n = dst.len();
        let mut i = 0;
        unsafe {
            while i + 4 <= n {
                let a = _mm_loadu_si128(dst.as_ptr().add(i) as *const __m128i);
                let b = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
                _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, _mm_add_epi32(a, b));
                i += 4;
            }
        }
        add_assign_u32_scalar(&mut dst[i..], &src[i..]);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: guarded by loop condition; NEON baseline.
        use std::arch::aarch64::*;
        let n = dst.len();
        let mut i = 0;
        unsafe {
            while i + 4 <= n {
                let a = vld1q_u32(dst.as_ptr().add(i));
                let b = vld1q_u32(src.as_ptr().add(i));
                vst1q_u32(dst.as_mut_ptr().add(i), vaddq_u32(a, b));
                i += 4;
            }
        }
        add_assign_u32_scalar(&mut dst[i..], &src[i..]);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        add_assign_u32_scalar(dst, src);
    }
}

fn add_assign_u64_scalar(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = d.wrapping_add(*s);
    }
}

fn add_assign_u64_dispatch(dst: &mut [u64], src: &[u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: guarded by loop condition; SSE2 baseline.
        use std::arch::x86_64::*;
        let n = dst.len();
        let mut i = 0;
        unsafe {
            while i + 2 <= n {
                let a = _mm_loadu_si128(dst.as_ptr().add(i) as *const __m128i);
                let b = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
                _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, _mm_add_epi64(a, b));
                i += 2;
            }
        }
        add_assign_u64_scalar(&mut dst[i..], &src[i..]);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: guarded by loop condition; NEON baseline.
        use std::arch::aarch64::*;
        let n = dst.len();
        let mut i = 0;
        unsafe {
            while i + 2 <= n {
                let a = vld1q_u64(dst.as_ptr().add(i));
                let b = vld1q_u64(src.as_ptr().add(i));
                vst1q_u64(dst.as_mut_ptr().add(i), vaddq_u64(a, b));
                i += 2;
            }
        }
        add_assign_u64_scalar(&mut dst[i..], &src[i..]);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        add_assign_u64_scalar(dst, src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::GrayImage;

    /// Deterministic LCG byte generator — stable across runs/targets.
    fn lcg_bytes(n: usize) -> Vec<u8> {
        let mut s = 0x1234_5678u32;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    fn lcg_image(w: usize, h: usize) -> GrayImage {
        GrayImage::from_vec(lcg_bytes(w * h), w, h)
    }

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

    // ------------------------------------------------------------------
    //  SIMD vs scalar equivalence (mandatory for every optimized kernel)
    // ------------------------------------------------------------------

    #[test]
    fn row_prefix_u32_simd_matches_scalar() {
        for n in [
            0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129,
            255, 256, 257, 1000, 1921,
        ] {
            let src = lcg_bytes(n);
            let mut scalar = vec![0u32; n];
            let mut simd = vec![0u32; n];
            row_prefix_u32_scalar(&src, &mut scalar, 0);
            row_prefix_u32_dispatch(&src, &mut simd);
            assert_eq!(scalar, simd, "u32 row prefix mismatch at len {n}");
        }
    }

    #[test]
    fn row_prefix_u32_scalar_carry_split() {
        // prefix(whole) == prefix(first half) ++ prefix(second half, carry).
        let src = lcg_bytes(77);
        let mut whole = vec![0u32; 77];
        row_prefix_u32_scalar(&src, &mut whole, 0);
        let mut second = vec![0u32; 40];
        row_prefix_u32_scalar(&src[37..], &mut second, whole[36]);
        assert_eq!(&whole[37..], &second[..]);
    }

    #[test]
    fn row_prefix_sq_u64_simd_matches_scalar() {
        for n in [
            0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 1000,
            1921,
        ] {
            let src = lcg_bytes(n);
            let mut scalar = vec![0u64; n];
            let mut simd = vec![0u64; n];
            row_prefix_sq_u64_scalar(&src, &mut scalar, 0);
            row_prefix_sq_u64_dispatch(&src, &mut simd);
            assert_eq!(scalar, simd, "u64 row prefix mismatch at len {n}");
        }
    }

    #[test]
    fn add_assign_simd_matches_scalar() {
        let n = 999;
        let a = lcg_bytes(n * 4);
        let b = lcg_bytes(n * 4);
        let mut u32a = vec![0u32; n];
        let mut u32b = vec![0u32; n];
        for i in 0..n {
            u32a[i] = u32::from_le_bytes([a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]]);
            u32b[i] = u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
        }
        let mut d1 = u32a.clone();
        let mut d2 = u32a.clone();
        add_assign_u32_scalar(&mut d1, &u32b);
        add_assign_u32_dispatch(&mut d2, &u32b);
        assert_eq!(d1, d2);

        let mut u64a = vec![0u64; n / 2 + 1];
        let mut u64b = vec![0u64; n / 2 + 1];
        for i in 0..u64a.len() {
            u64a[i] = (i as u64).wrapping_mul(0x9E37_79B9) | 1;
            u64b[i] = (i as u64).wrapping_mul(i as u64) | 1;
        }
        let mut e1 = u64a.clone();
        let mut e2 = u64a.clone();
        add_assign_u64_scalar(&mut e1, &u64b);
        add_assign_u64_dispatch(&mut e2, &u64b);
        assert_eq!(e1, e2);
    }

    /// Naive O(W·H) rectangle-sum reference for `IntegralImage`.
    fn naive_integral_data(img: &GrayImage) -> Vec<u32> {
        let w = img.width();
        let h = img.height();
        let stride = w + 1;
        let mut d = vec![0u32; stride * (h + 1)];
        for y in 1..=h {
            for x in 1..=w {
                d[y * stride + x] =
                    img[(x - 1, y - 1)] as u32 + d[(y - 1) * stride + x] + d[y * stride + x - 1]
                        - d[(y - 1) * stride + x - 1];
            }
        }
        d
    }

    #[test]
    fn from_gray_matches_naive_reference() {
        for (w, h) in [
            (1usize, 1usize),
            (1, 9),
            (3, 3),
            (15, 7),
            (16, 16),
            (17, 31),
            (64, 65),
        ] {
            let img = lcg_image(w, h);
            let ii = IntegralImage::from_gray(&img);
            assert_eq!(ii.data, naive_integral_data(&img), "mismatch at {w}x{h}");
            // Spot-check rectangle sums against direct summation.
            for &(x1, y1, x2, y2) in &[(0, 0, w, h), (1, 1, w - 1, h - 1), (0, 0, 1, 1)] {
                let expect: u64 = (y1..y2)
                    .flat_map(|y| (x1..x2).map(move |x| (x, y)))
                    .map(|(x, y)| img[(x, y)] as u64)
                    .sum();
                assert_eq!(ii.rect_sum(x1, y1, x2, y2), expect);
            }
        }
    }

    #[test]
    fn from_gray_sq_matches_naive_reference() {
        for (w, h) in [
            (1usize, 1usize),
            (3, 5),
            (16, 16),
            (17, 31),
            (64, 65),
            (100, 3),
        ] {
            let img = lcg_image(w, h);
            let sq = SquaredIntegralImage::from_gray(&img);
            let w_ = img.width();
            let h_ = img.height();
            let stride = w_ + 1;
            let mut naive = vec![0u64; stride * (h_ + 1)];
            for y in 1..=h_ {
                for x in 1..=w_ {
                    let v = img[(x - 1, y - 1)] as u64;
                    naive[y * stride + x] =
                        v * v + naive[(y - 1) * stride + x] + naive[y * stride + x - 1]
                            - naive[(y - 1) * stride + x - 1];
                }
            }
            assert_eq!(sq.data, naive, "squared mismatch at {w}x{h}");
        }
    }

    /// Reference implementation of the *historic* two-pass rotated integral
    /// (kept verbatim from the pre-optimisation code) to prove the single-pass
    /// diagonal recurrence is bit-identical.
    fn rotated_reference(img: &GrayImage) -> Vec<i64> {
        let w = img.width();
        let h = img.height();
        let stride = w + 1;
        let mut s = vec![0i64; stride * (h + 1)];
        for y in 1..=h {
            let mut row_acc: i64 = 0;
            for x in 1..=w {
                row_acc += img[(x - 1, y - 1)] as i64;
                s[y * stride + x] = row_acc + s[(y - 1) * stride + x];
            }
        }
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
        data
    }

    #[test]
    fn rotated_single_pass_matches_two_pass_reference() {
        for (w, h) in [
            (1usize, 1usize),
            (2, 2),
            (3, 3),
            (5, 3),
            (17, 9),
            (31, 33),
            (64, 64),
            (65, 3),
        ] {
            let img = lcg_image(w, h);
            let ri = RotatedIntegralImage::from_gray(&img);
            assert_eq!(ri.data, rotated_reference(&img), "mismatch at {w}x{h}");
        }
    }

    #[test]
    fn unchecked_sums_match_checked() {
        let (w, h) = (13usize, 9usize);
        let img = lcg_image(w, h);
        let ii = IntegralImage::from_gray(&img);
        let sq = SquaredIntegralImage::from_gray(&img);
        let ri = RotatedIntegralImage::from_gray(&img);
        for y1 in 0..h {
            for y2 in (y1 + 1)..=h {
                for x1 in 0..w {
                    for x2 in (x1 + 1)..=w {
                        assert_eq!(
                            ii.rect_sum(x1, y1, x2, y2),
                            ii.rect_sum_unchecked(x1, y1, x2, y2),
                            "rect_sum mismatch at ({x1},{y1},{x2},{y2})"
                        );
                        assert_eq!(
                            sq.rect_sum_sq(x1, y1, x2, y2),
                            sq.rect_sum_sq_unchecked(x1, y1, x2, y2)
                        );
                        assert_eq!(
                            ii.tilted_rect_sum(&ri, x1, y1, x2, y2),
                            ri.tilted_rect_sum_unchecked(x1, y1, x2, y2)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rotated_empty_is_safe() {
        let ri = RotatedIntegralImage::empty();
        // Empty 0×0 table: every non-empty rect is clamped away to 0 before
        // any cell is touched; degenerate rects short-circuit to 0 too.
        assert_eq!(ri.tilted_rect_sum(0, 0, 4, 4), 0);
        assert_eq!(ri.tilted_rect_sum(2, 2, 3, 3), 0);
        assert_eq!(ri.tilted_rect_sum(5, 5, 5, 5), 0);
    }

    #[test]
    fn integral_buffers_recycle_through_pool() {
        crate::pool::clear();
        let img = lcg_image(20, 14);
        let p1 = {
            let ii = IntegralImage::from_gray(&img);
            ii.data.as_ptr()
        };
        let ii2 = IntegralImage::from_gray(&img);
        // After dropping the first image its buffer returns to the pool and
        // the next same-size build reuses it (same backing allocation).
        assert_eq!(p1, ii2.data.as_ptr());
        // Same for the squared integral.
        let q1 = {
            let sq = SquaredIntegralImage::from_gray(&img);
            sq.data.as_ptr()
        };
        let sq2 = SquaredIntegralImage::from_gray(&img);
        assert_eq!(q1, sq2.data.as_ptr());
    }

    /// A dirty recycled buffer must not leak into the table: poison the
    /// pool with 0xFF-filled buffers, then build — every cell must still
    /// match the naive reference (padding cells re-zeroed, all others
    /// rewritten).
    #[test]
    fn integral_survives_dirty_pooled_buffer() {
        crate::pool::clear();
        let img = lcg_image(23, 7);
        // Poison: build once, drop (returns buffer to pool), then manually
        // dirty the pooled buffer through acquire/release.
        {
            let _ = IntegralImage::from_gray(&img);
        }
        {
            let mut v = crate::pool::acquire_integral(23, 7);
            v.fill(u32::MAX);
            crate::pool::release_integral(23, 7, v);
        }
        {
            let _ = SquaredIntegralImage::from_gray(&img);
        }
        {
            let mut v = crate::pool::acquire_integral_u64(23, 7);
            v.fill(u64::MAX);
            crate::pool::release_integral_u64(23, 7, v);
        }
        let ii = IntegralImage::from_gray(&img);
        assert_eq!(ii.data, naive_integral_data(&img));
        let sq = SquaredIntegralImage::from_gray(&img);
        let w_ = img.width();
        let h_ = img.height();
        let stride = w_ + 1;
        let mut naive = vec![0u64; stride * (h_ + 1)];
        for y in 1..=h_ {
            for x in 1..=w_ {
                let v = img[(x - 1, y - 1)] as u64;
                naive[y * stride + x] =
                    v * v + naive[(y - 1) * stride + x] + naive[y * stride + x - 1]
                        - naive[(y - 1) * stride + x - 1];
            }
        }
        assert_eq!(sq.data, naive);
    }

    #[test]
    fn passes_variance_sums_matches_passes_variance() {
        let img = lcg_image(40, 30);
        let ii = IntegralImage::from_gray(&img);
        let sq = SquaredIntegralImage::from_gray(&img);
        for x in [0usize, 3, 7] {
            for y in [0usize, 2, 11] {
                for thr in [0u64, 200, 10_000, 1 << 40] {
                    let sum = ii.rect_sum(x, y, x + 24, y + 24);
                    let sum_sq = sq.rect_sum_sq(x, y, x + 24, y + 24);
                    assert_eq!(
                        sq.passes_variance(&ii, x, y, 24, 24, thr),
                        SquaredIntegralImage::passes_variance_sums(sum, sum_sq, 24, 24, thr),
                        "mismatch at ({x},{y}) thr={thr}"
                    );
                }
            }
        }
    }
}
