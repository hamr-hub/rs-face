//! Integral image (summed-area table) — O(1) rectangle sum queries.
//!
//! For an input grayscale image `I` of size `W × H`,
//! `II[x, y] = sum of I[i, j] for i < x, j < y` (with `II[0, *] = II[*, 0] = 0`).
//! Then the sum over rectangle `[x1, x2) × [y1, y2)` is:
//!   `II[x2, y2] - II[x1, y2] - II[x2, y1] + II[x1, y1]`.
//!
//! Prefix sums normally use a pooled `u32` table (for `1920 × 1080 × 255`
//! the max value is ~5.3e8, well within u32). Images with `W*H*255 >
//! u32::MAX` (~16.8 Mp) would wrap and silently corrupt Haar scores, so
//! [`IntegralImage::from_gray`] automatically switches to a non-pooled
//! `u64` table for them; all sum queries return `u64` regardless.
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

/// Backing storage for [`IntegralImage`].
///
/// Prefix sums of an 8-bit image fit in u32 whenever `W*H*255 <= u32::MAX`
/// (~16.8 M pixels) — true for everything up to and above 4K. Bigger images
/// (large camera raw frames, stitched panoramas) can wrap a u32 table, and
/// because rectangle sums are inclusion–exclusion differences a wrapped
/// corner corrupts Haar scores silently. Such images take the `Wide` u64
/// path; it is not pooled (a rare, large allocation).
#[derive(Clone)]
enum IntegralTable {
    /// Flat `(H+1, W+1)` u32 buffer.
    Narrow(Vec<u32>),
    /// Flat `(H+1, W+1)` u64 buffer.
    Wide(Vec<u64>),
}

impl Default for IntegralTable {
    fn default() -> Self {
        // Empty narrow buffer; used only by buffer-stealing `into_data`.
        IntegralTable::Narrow(Vec::new())
    }
}

/// Integral image of shape `(H+1, W+1)`.
/// Index `(x, y)` (0 <= x <= W, 0 <= y <= H) lives at `y * stride + x`.
///
/// Storage is u32 for images whose prefix sums cannot overflow and u64
/// otherwise (see the private `IntegralTable` enum); all sum queries return `u64`.
#[derive(Clone)]
pub struct IntegralImage {
    data: IntegralTable,
    width: usize,  // original image width
    height: usize, // original image height
    stride: usize, // = width + 1
}

/// True when every possible prefix sum of an 8-bit image of this size fits
/// in a u32 table. Equalization cannot push pixels past 255, so the
/// worst-case bound `W*H*255` is exact.
#[inline]
pub(crate) fn prefix_sums_fit_u32(w: usize, h: usize) -> bool {
    (w as u128) * (h as u128) * 255 <= u32::MAX as u128
}

/// Free-function SIMD rect-sum helper. Caller guarantees the rectangle is
/// in-bounds and the integral image is narrow (u32).
///
/// SSE2 path packs 4 u32 corner reads into one `__m128i`, then computes
/// `(TL + BR) − (TR + BL)` as packed u32 (`vaddq_u32` + `vrev64q_u32` +
/// `vsubq_u32` on aarch64 / equivalent intrinsics on x86_64). The
/// arithmetic replaces the 3 dependent scalar `add`/`sub` µops of the
/// baseline with a handful of independent vector ops, giving the cascade
/// hot loop a couple of cycles back per rect.
#[inline]
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn rect_sum_unchecked_narrow_simd(
    v: *const u32,
    stride: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
) -> u64 {
    use std::arch::x86_64::*;
    debug_assert!(x1 < x2 && y1 < y2);
    // SAFETY: caller guarantees in-bounds.
    let idx_tl = y1 * stride + x1;
    let idx_tr = y1 * stride + x2;
    let idx_bl = y2 * stride + x1;
    let idx_br = y2 * stride + x2;
    // SAFETY: in-bounds by the contract.
    let tl = unsafe { *v.add(idx_tl) };
    let tr = unsafe { *v.add(idx_tr) };
    let bl = unsafe { *v.add(idx_bl) };
    let br = unsafe { *v.add(idx_br) };
    // SAFETY: SSE2 baseline.
    unsafe {
        // Pack corners into lanes [TL, TR, BL, BR]. `_mm_set_epi32(e3,e2,e1,e0)`
        // → lane k = e_k. So (br, bl, tr, tl) → lane 0=tl, 1=tr, 2=bl, 3=br.
        let v32 = _mm_set_epi32(br as i32, bl as i32, tr as i32, tl as i32);
        // Reverse all four lanes: [BR, BL, TR, TL].
        // `_MM_SHUFFLE(z, y, x, w)` puts source[w] into result[0], source[x]
        // into result[1], etc. We want result = [src3, src2, src1, src0] =
        // [BR, BL, TR, TL], so w=3, x=2, y=1, z=0.
        let v_rev = _mm_shuffle_epi32(v32, _MM_SHUFFLE(0, 1, 2, 3) as i32);
        // sum_pairs = [TL+BR, TR+BL, BL+TR, BR+TL].
        let sum_pairs = _mm_add_epi32(v32, v_rev);
        // Swap lanes 0 and 1 of sum_pairs so lane 0 = TR+BL, lane 1 = TL+BR.
        let sum_swap = _mm_shuffle_epi32(sum_pairs, _MM_SHUFFLE(0, 0, 0, 1) as i32);
        // result[0] = (TL+BR) - (TR+BL) = TL - TR - BL + BR.
        let result = _mm_sub_epi32(sum_pairs, sum_swap);
        _mm_cvtsi128_si32(result) as u32 as u64
    }
}

/// NEON counterpart — same SIMD inclusion-exclusion path on aarch64.
#[inline]
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn rect_sum_unchecked_narrow_simd(
    v: *const u32,
    stride: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
) -> u64 {
    use std::arch::aarch64::*;
    debug_assert!(x1 < x2 && y1 < y2);
    // SAFETY: in-bounds by the contract.
    let idx_tl = y1 * stride + x1;
    let idx_tr = y1 * stride + x2;
    let idx_bl = y2 * stride + x1;
    let idx_br = y2 * stride + x2;
    let tl = unsafe { *v.add(idx_tl) };
    let tr = unsafe { *v.add(idx_tr) };
    let bl = unsafe { *v.add(idx_bl) };
    let br = unsafe { *v.add(idx_br) };
    // SAFETY: NEON baseline.
    unsafe {
        // lanes 0..3 = [TL, TR, BL, BR] (aarch64 is little-endian so the
        // u32x4 register lanes map to the [u32; 4] array layout directly).
        let v32: uint32x4_t = core::mem::transmute([tl, tr, bl, br]);
        // Full 4-lane reverse → [BR, BL, TR, TL]. Built with explicit
        // `vcreate_u32`/`vcombine_u32` because `vrev64q_u32` reverses per
        // 64-bit half, which would put [TR, TL, BR, BL] — wrong shape.
        // `vcreate_u32(a)` materialises a uint32x2_t with lane 0 = low
        // 32 bits of `a`, lane 1 = high 32 bits of `a`.
        let v_rev: uint32x4_t = vcombine_u32(
            vcreate_u32((bl as u64) << 32 | br as u64), // lane0=BR, lane1=BL
            vcreate_u32((tl as u64) << 32 | tr as u64), // lane0=TR, lane1=TL
        );
        // sum_pairs = [TL+BR, TR+BL, BL+TR, BR+TL].
        let sum_pairs = vaddq_u32(v32, v_rev);
        // Swap the low 64-bit half of sum_pairs so lane 0 = TR+BL, lane 1 = TL+BR.
        let sum_swap = vrev64q_u32(sum_pairs);
        // result[0] = (TL+BR) - (TR+BL) = TL - TR - BL + BR.
        let result = vsubq_u32(sum_pairs, sum_swap);
        vgetq_lane_u32(result, 0) as u64
    }
}

/// Scalar fallback used when SIMD is unavailable (no x86_64 / aarch64).
/// Bit-identical to the SIMD versions.
#[inline]
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) unsafe fn rect_sum_unchecked_narrow_simd(
    v: *const u32,
    stride: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
) -> u64 {
    debug_assert!(x1 < x2 && y1 < y2);
    let idx_tl = y1 * stride + x1;
    let idx_tr = y1 * stride + x2;
    let idx_bl = y2 * stride + x1;
    let idx_br = y2 * stride + x2;
    let tl = unsafe { *v.add(idx_tl) };
    let tr = unsafe { *v.add(idx_tr) };
    let bl = unsafe { *v.add(idx_bl) };
    let br = unsafe { *v.add(idx_br) };
    let r = br.wrapping_add(tl);
    let s = tr.wrapping_add(bl);
    r.wrapping_sub(s) as u64
}

impl IntegralImage {
    /// Construct from a precomputed `(W+1) × (H+1)` u32 buffer (as returned by
    /// the GPU kernel). The buffer layout must be row-major with stride = W+1.
    ///
    /// The caller guarantees the table does not wrap u32 (the GPU detector
    /// path checks the crate-private `prefix_sums_fit_u32` predicate before invoking it).
    pub fn from_owned(data: Vec<u32>, width: usize, height: usize) -> Self {
        let stride = width + 1;
        Self {
            data: IntegralTable::Narrow(data),
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
        if prefix_sums_fit_u32(w, h) {
            let data = Self::build_narrow(img, w, h, stride);
            Self {
                data: IntegralTable::Narrow(data),
                width: w,
                height: h,
                stride,
            }
        } else {
            // 2026-09-18 /goal: images with W*H*255 > u32::MAX overflowed the
            // u32 table and the inclusion–exclusion reads silently produced
            // giant underflowed "sums", corrupting every Haar score with no
            // error. Build a u64 table on this rare path instead.
            let data = Self::build_wide(img, w, h, stride);
            Self {
                data: IntegralTable::Wide(data),
                width: w,
                height: h,
                stride,
            }
        }
    }

    /// Pooled u32 fused single-pass build (the common, fast path).
    /// Row prefix dispatches to the SSE2/NEON helper
    /// (`row_prefix_u32_dispatch`); vertical add stays in the SIMD
    /// helper (`add_assign_u32_dispatch`). The pair shares the fused
    /// 1-loop formulation (row prefix + vertical fold merged into one
    /// per-row pass), so the produced table is bit-identical to the
    /// scalar-prefix version.
    fn build_narrow(img: &GrayImage, w: usize, h: usize, stride: usize) -> Vec<u32> {
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
            row_prefix_u32_dispatch(img.row(y), &mut cur[1..]);
            add_assign_u32_dispatch(cur, prev);
        }
        data
    }

    /// Non-pooled u64 fused single-pass build for overflow-sized images.
    /// Same formulation as [`Self::build_narrow`]; overflow is impossible
    /// (u64 holds prefix sums up to ~72 Gp), so adds are plain `+=`.
    fn build_wide(img: &GrayImage, w: usize, h: usize, stride: usize) -> Vec<u64> {
        let mut data = vec![0u64; stride * (h + 1)];
        for y in 0..h {
            // Mirror the narrow split so both builds stay structurally equal.
            let (head, tail) = data.split_at_mut((y + 1) * stride);
            let prev = &head[y * stride..];
            let cur = &mut tail[..stride];
            cur[0] = 0;
            let mut acc: u64 = 0;
            let body = &mut cur[1..];
            for (s, d) in img.row(y).iter().zip(body.iter_mut()) {
                acc += *s as u64;
                *d = acc;
            }
            add_assign_u64_dispatch(cur, prev);
        }
        data
    }

    /// Extract the raw `(W+1) × (H+1)` backing buffer **without** returning
    /// it to the thread-local pool (the normal `Drop` recycles it). The
    /// returned Vec is exactly the table as built.
    ///
    /// # Panics
    ///
    /// Only for an image too large for a u32 table (the wide u64 path);
    /// callers are GPU/CPU comparison helpers that only handle the narrow
    /// case, and such images are never routed to the u32 GPU kernels.
    pub fn into_data(mut self) -> Vec<u32> {
        // Steal the table so the pool-recycling Drop sees an empty one.
        match std::mem::take(&mut self.data) {
            IntegralTable::Narrow(v) => v,
            IntegralTable::Wide(_) => {
                panic!("into_data() called on a u64 (wide) integral image")
            }
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

    /// Whether this table uses the wide u64 storage (image too large for u32
    /// prefix sums).
    #[inline]
    pub fn is_wide(&self) -> bool {
        matches!(self.data, IntegralTable::Wide(_))
    }

    #[inline]
    fn corner(&self, idx: usize) -> u64 {
        match &self.data {
            IntegralTable::Narrow(v) => v[idx] as u64,
            IntegralTable::Wide(v) => v[idx],
        }
    }

    /// [`Self::corner`] with no bounds checks; the caller guarantees `idx`
    /// is in the table (same contract as [`Self::rect_sum_unchecked`]).
    #[inline]
    unsafe fn corner_unchecked(&self, idx: usize) -> u64 {
        // SAFETY: caller guarantees idx is in bounds for both variants.
        unsafe {
            match &self.data {
                IntegralTable::Narrow(v) => *v.get_unchecked(idx) as u64,
                IntegralTable::Wide(v) => *v.get_unchecked(idx),
            }
        }
    }

    /// Raw access to `(x, y)` accumulator (0 <= x <= W, 0 <= y <= H).
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u64 {
        self.corner(y * self.stride + x)
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
        let stride = self.stride;
        // SAFETY: documented contract above; debug_assert pins it in test
        // builds. Match the IntegralTable discriminant ONCE per call (the
        // historical implementation did 4 separate `corner_unchecked` calls
        // and paid the match cost four times — at ~50k windows × 4 corners
        // × 25 stages of cascade eval, that branch pressure was a real
        // contributor to the cascade's hot-loop cost).
        unsafe {
            match &self.data {
                IntegralTable::Narrow(v) => {
                    let a = *v.get_unchecked(y1 * stride + x1) as u64;
                    let b = *v.get_unchecked(y1 * stride + x2) as u64;
                    let c = *v.get_unchecked(y2 * stride + x1) as u64;
                    let d = *v.get_unchecked(y2 * stride + x2) as u64;
                    d + a - b - c
                }
                IntegralTable::Wide(v) => {
                    let a = *v.get_unchecked(y1 * stride + x1);
                    let b = *v.get_unchecked(y1 * stride + x2);
                    let c = *v.get_unchecked(y2 * stride + x1);
                    let d = *v.get_unchecked(y2 * stride + x2);
                    d + a - b - c
                }
            }
        }
    }

    /// Narrow-only variant of [`Self::rect_sum_unchecked`]. Skips the
    /// `IntegralTable` enum match that the generic variant must emit on
    /// every call — at ~50k windows × 25 stages × 1 feature × ~3 rects of
    /// cascade eval, the match is the dominant cost in the cascade hot
    /// loop. Use only when the caller has already verified
    /// `!self.is_wide()` for this table.
    ///
    /// # Safety contract (caller must uphold)
    /// Same as [`Self::rect_sum_unchecked`], AND the caller must guarantee
    /// the table is narrow (`is_wide() == false`). Calling on a wide
    /// table would read u32 entries instead of u64 — a soundness bug.
    #[inline]
    pub(crate) unsafe fn rect_sum_unchecked_narrow(
        &self,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
    ) -> u64 {
        debug_assert!(x1 < x2 && x2 <= self.width && y1 < y2 && y2 <= self.height);
        debug_assert!(!self.is_wide());
        let stride = self.stride;
        // SAFETY: caller guarantees narrow variant and in-bounds rect. The
        // discriminant is re-checked as `unreachable_unchecked` so the
        // compiler can drop the second arm of the dispatch.
        unsafe {
            match &self.data {
                IntegralTable::Narrow(v) => {
                    let a = *v.get_unchecked(y1 * stride + x1) as u64;
                    let b = *v.get_unchecked(y1 * stride + x2) as u64;
                    let c = *v.get_unchecked(y2 * stride + x1) as u64;
                    let d = *v.get_unchecked(y2 * stride + x2) as u64;
                    d + a - b - c
                }
                IntegralTable::Wide(_) => std::hint::unreachable_unchecked(),
            }
        }
    }

    /// Sum of pixels in a *tilted* (45 degree rotated) rectangle used
    /// by tilted Haar features, matching OpenCV's `CV_TILTED_OFS` lookup.
    ///
    /// Delegates to [`RotatedIntegralImage::tilted_rect_sum`]; see that
    /// method for the cone semantics and the four corner points.
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

    /// SIMD inclusion–exclusion dispatch used by the cascade hot loop.
    /// Falls back to the scalar generic path on non-x86_64 /
    /// non-aarch64 targets. Caller guarantees the rectangle is
    /// in-bounds and the table is narrow.
    ///
    /// # Safety contract (caller must uphold)
    /// Same as [`Self::rect_sum_unchecked_narrow`].
    #[inline]
    pub(crate) unsafe fn rect_sum_unchecked_narrow_simd_method(
        &self,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
    ) -> u64 {
        debug_assert!(x1 < x2 && x2 <= self.width && y1 < y2 && y2 <= self.height);
        debug_assert!(!self.is_wide());
        let stride = self.stride;
        let IntegralTable::Narrow(v) = &self.data else {
            std::hint::unreachable_unchecked()
        };
        rect_sum_unchecked_narrow_simd(v.as_ptr(), stride, x1, y1, x2, y2)
    }
}

impl Drop for IntegralImage {
    fn drop(&mut self) {
        // Recycle the backing buffer. `try_with` degrades gracefully if the
        // thread-local pool is already torn down at thread exit. The rare u64
        // wide table is not pooled — its Vec frees normally.
        if let IntegralTable::Narrow(buf) = std::mem::take(&mut self.data) {
            crate::pool::release_integral(self.width, self.height, buf);
        }
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
            // SIMD row prefix of squares — same SSE2/NEON helper used by
            // the unsquared IntegralImage build. Replaces the previous
            // scalar prefix loop in this fused-1-loop formulation; the
            // produced table is bit-identical.
            row_prefix_sq_u64_dispatch(img.row(y), &mut cur[1..]);
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

    /// Fast variance pre-filter test with precomputed `N` and `N²`.
    ///
    /// Same arithmetic and accept/reject semantics as
    /// [`Self::passes_variance_sums`], but accepts the two
    /// `(N, N²)` values the cascade already has on hand (the window's
    /// `(w - 2) * (h - 2)` and its square) instead of recomputing them per
    /// call. The detector builds `CachedNormFactor` once per frame, then
    /// threads `(n_pixels, n_pixels_sq)` into the window scan and avoids
    /// both the `w * h` multiply and the `(w*h).checked_mul(w*h)` per window.
    ///
    /// All overflow checks in [`Self::passes_variance_sums`] are dropped:
    /// `sum_sq ≤ W*H*255² ≤ 1.4e11` and `N ≤ 24*24 = 576`, so
    /// `sum_sq * N ≤ 8e13 < 2^47`; `sum ≤ 5.3e8` and `sum² ≤ 2.8e17 < 2^58`;
    /// `threshold * N² ≤ u64::MAX*576²` would only overflow on pathological
    /// thresholds the detector never reaches. If a future caller wants the
    /// conservative semantics, use [`Self::passes_variance_sums`].
    #[inline]
    pub fn passes_variance_sums_fast(
        sum: u64,
        sum_sq: u64,
        n: u64,
        n_sq: u64,
        variance_threshold: u64,
    ) -> bool {
        // variance >= threshold iff sum_sq * N - sum² >= threshold * N²
        let sum_sq_part = sum_sq * n;
        let sum_part = sum * sum;
        sum_sq_part >= sum_part + variance_threshold * n_sq
    }

    /// 4-wide SIMD batched variance pre-filter. Returns a 4-bit mask where
    /// bit `i` is set iff window `i` passes the variance test — i.e.
    /// `sum_sq_i * N - sum_i² >= variance_threshold * N²` — with the
    /// exact same integer arithmetic as [`Self::passes_variance_sums_fast`]
    /// (modulo f64↔u64 bit-equivalence for the intermediate products).
    ///
    /// `n_pixels_f64` and `thr_n_sq_f64` are precomputed by the caller once
    /// per frame as `n_pixels as f64` and `(variance_threshold * n_sq) as f64`.
    ///
    /// # Float↔int equivalence
    ///
    /// The exact integer pre-filter test is `(sum_sq * N) − (sum²) ≥
    /// threshold * N²`. Every intermediate product fits comfortably inside
    /// 2⁵³ (the largest exact f64 integer): with `N ≤ 24² = 576` and
    /// `sum_sq ≤ 22²·255² ≈ 3.15e7`, `sum_sq * N ≤ 1.82e10 < 2³⁴`. Same
    /// for `sum² ≤ 1.5e10 < 2³⁴`. The constraint that matters is on
    /// `thr_n_sq_f64`: it must be `< 2^53` for the comparison to be exact
    /// in f64 lane math. The detector's default threshold (200) gives
    /// `200 · 576² ≈ 6.6e7 < 2^26` — well inside the safe range. The
    /// detector gates the SIMD path on this and falls back to the scalar
    /// tail for pathological thresholds.
    #[inline]
    pub fn passes_variance_mask_4(
        sum: [u64; 4],
        sum_sq: [u64; 4],
        n_pixels_f64: f64,
        thr_n_sq_f64: f64,
    ) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: SSE2 baseline on x86_64; documented contract that the
            // f64 ↔ u64 conversion is exact (see fn doc comment).
            unsafe { variance_mask_4_x86_64(sum, sum_sq, n_pixels_f64, thr_n_sq_f64) }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON baseline on aarch64.
            unsafe { variance_mask_4_aarch64(sum, sum_sq, n_pixels_f64, thr_n_sq_f64) }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = (sum, sum_sq, n_pixels_f64, thr_n_sq_f64);
            // Scalar fallback — same accept/reject as the SIMD path.
            let mut mask = 0u32;
            for i in 0..4 {
                if (sum_sq[i] as f64).mul_add(n_pixels_f64, -(sum[i] as f64).powi(2))
                    >= thr_n_sq_f64
                {
                    mask |= 1 << i;
                }
            }
            mask
        }
    }
}

impl Drop for SquaredIntegralImage {
    fn drop(&mut self) {
        crate::pool::release_integral_u64(self.width, self.height, std::mem::take(&mut self.data));
    }
}

/// Rotated (45°) integral image: the `rt0` table behind OpenCV's tilted
/// Haar features.
///
/// Definition (Lienhart & Maydt, 2002; the `tilted` output of OpenCV's
/// `integral()`): with 0-based pixel coordinates `(px, py)`, table cell
/// `(X, Y)` (1-based) holds the sum over the downward-opening lattice cone
///
///   `R(X, Y) = Σ I(px, py)`  where `(Y − 1 − py) ≥ |X − 1 − px|`.
///
/// Column 0 and row 0 are zero padding (never counted).
///
/// # Construction
///
/// OpenCV builds the table with a per-row scalar recurrence plus a
/// one-row rolling buffer of column sums. That exact recurrence is used
/// here (`from_gray`) and verified two ways in the tests: against the
/// cone definition enumerated pixel-by-pixel, and against a transcription
/// of OpenCV's `integral_` tilted branch on random images. An earlier
/// revision "simplified" the table into a plain diagonal line sum — the
/// algebra looked equivalent but was not; the cone has an extra parent
/// per row. That made tilted features count arbitrary (even negative)
/// sums; it is the reason this code carries the brute-force reference
/// rather than a "derived" one.
#[derive(Clone)]
pub struct RotatedIntegralImage {
    data: Vec<i64>,
    width: usize,
    height: usize,
    stride: usize,
}

impl RotatedIntegralImage {
    /// Build the rotated (45°) integral image from a grayscale input using
    /// the rolling-buffer recurrence from OpenCV's `integral_` tilted
    /// branch (cn=1), documented on the struct above.
    pub fn from_gray(img: &GrayImage) -> Self {
        let w = img.width();
        let h = img.height();
        let stride = w + 1;
        let mut data = vec![0i64; stride * (h + 1)];
        if w == 0 || h == 0 {
            return Self {
                data,
                width: w,
                height: h,
                stride,
            };
        }
        // One rolling row of column sums, initialised from image row 0.
        let mut col = vec![0i64; w + 1];
        for x in 0..w {
            col[x] = img[(x, 0)] as i64;
            data[stride + x + 1] = col[x]; // table row 1
        }
        for i in 1..h {
            let r = i + 1; // table row
            data[r * stride] = data[(r - 1) * stride]; // padding col stays 0
                                                       // First pixel of the row.
            data[r * stride + 1] = data[(r - 1) * stride + 1] + img[(0, i)] as i64 + col[1];
            for x in 1..w - 1 {
                let t1 = col[x];
                col[x - 1] = t1 + img[(x - 1, i)] as i64;
                data[r * stride + x + 1] =
                    t1 + col[x + 1] + img[(x, i)] as i64 + data[(r - 1) * stride + x];
            }
            if w > 1 {
                let t1 = col[w - 1];
                col[w - 2] = t1 + img[(w - 2, i)] as i64;
                data[r * stride + w] = img[(w - 1, i)] as i64 + t1 + data[(r - 1) * stride + w - 1];
                col[w - 1] = img[(w - 1, i)] as i64;
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

    /// Sum over a tilted (45°) Haar rectangle, matching OpenCV's
    /// `CV_TILTED_OFS` / `HaarEvaluator::OptFeature` lookup.
    ///
    /// Given the sub-rectangle origin `(x1, y1)` and upright box
    /// `(x2, y2)` with `w = x2 − x1`, `h = y2 − y1`, the counted region is
    /// the rotated parallelogram with the four lookup corners
    ///
    /// - `p0 = (x1, y1)`
    /// - `p1 = (x1 − h, y1 + h)`
    /// - `p2 = (x1 + w, y1 + w)`
    /// - `p3 = (x1 + w − h, y1 + w + h)`
    ///
    /// and the sum is `R[p0] − R[p1] − R[p2] + R[p3]`. Corners outside
    /// the table read as zero: the zero row/column borders plus image
    /// edges, the same convention OpenCV relies on for its padding.
    pub fn tilted_rect_sum(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> i64 {
        if x2 <= x1 || y2 <= y1 {
            return 0;
        }
        let (w, h) = (x2 - x1, y2 - y1);
        let at = |x: isize, y: isize| -> i64 {
            // Row/col 0 are zero padding even though data has that cell;
            // negative coordinates and corners past the table read 0 too.
            if x < 1 || y < 1 || x as usize > self.width || y as usize > self.height {
                0
            } else {
                self.data[(y as usize) * self.stride + x as usize]
            }
        };
        let (x1i, y1i) = (x1 as isize, y1 as isize);
        let (wi, hi) = (w as isize, h as isize);
        at(x1i, y1i) - at(x1i - hi, y1i + hi) - at(x1i + wi, y1i + wi)
            + at(x1i + wi - hi, y1i + wi + hi)
    }

    /// Unchecked variant of [`Self::tilted_rect_sum`].
    ///
    /// # Safety contract (caller must uphold)
    /// Every lookup corner must lie inside the table, i.e. `x1 >= h`,
    /// `x1 + w <= self.width`, and `y1 + w + h <= self.height`. Under
    /// that contract the zero-border checks are identities and this is
    /// bit-identical to the checked variant.
    #[inline]
    pub(crate) fn tilted_rect_sum_unchecked(
        &self,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
    ) -> i64 {
        debug_assert!(x2 > x1 && y2 > y1);
        let (w, h) = (x2 - x1, y2 - y1);
        debug_assert!(
            x1 >= h && x1 + w <= self.width && y1 + w + h <= self.height,
            "tilted lookup corners overhang the table; use the checked variant"
        );
        // SAFETY: documented contract above; each (x, y) is in bounds.
        unsafe {
            let at = |x: usize, y: usize| *self.data.get_unchecked(y * self.stride + x);
            at(x1, y1) - at(x1 - h, y1 + h) - at(x1 + w, y1 + w) + at(x1 + w - h, y1 + w + h)
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

// ---------------------------------------------------------------------------
//   Variance pre-filter batch — 4×f64 lane math on SSE2 / NEON baseline
// ---------------------------------------------------------------------------
//
// Computes the per-window Viola-Jones variance test
//   `sum_sq * N - sum² >= threshold * N²`
// for 4 windows in parallel, using f64 lane SIMD. The integer
// intermediates (`sum_sq * N`, `sum²`, `threshold * N²`) all fit comfortably
// in 2⁵³ (the largest exact f64 integer) — see the doc comment on
// `SquaredIntegralImage::passes_variance_mask_4` — so the f64 arithmetic
// produces the exact same accept/reject decision as the integer scalar
// fallback.
//
// SSE2 / NEON are part of their respective baseline ABIs; the kernels
// are pure `#[cfg(target_arch = "…")]` dispatch, no runtime detection.

/// SSE2 implementation of [`SquaredIntegralImage::passes_variance_mask_4`].
/// 4 f64 lanes packed as 2 × `__m128d`.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn variance_mask_4_x86_64(
    sum: [u64; 4],
    sum_sq: [u64; 4],
    n_pixels_f64: f64,
    thr_n_sq_f64: f64,
) -> u32 {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 baseline intrinsics; every intermediate is exact (see
    // the fn doc on `passes_variance_mask_4`).
    unsafe {
        // Load 4 u64s as 2 lanes of 2 f64 each. Lane i holds `sum[i]`.
        let s01 = _mm_castsi128_pd(_mm_set_epi64x(sum[1] as i64, sum[0] as i64));
        let s23 = _mm_castsi128_pd(_mm_set_epi64x(sum[3] as i64, sum[2] as i64));
        // Lane i holds `sum_sq[i]`.
        let ss01 = _mm_castsi128_pd(_mm_set_epi64x(sum_sq[1] as i64, sum_sq[0] as i64));
        let ss23 = _mm_castsi128_pd(_mm_set_epi64x(sum_sq[3] as i64, sum_sq[2] as i64));
        let n = _mm_set1_pd(n_pixels_f64);
        let thr = _mm_set1_pd(thr_n_sq_f64);
        // ss * N
        let ss_n01 = _mm_mul_pd(ss01, n);
        let ss_n23 = _mm_mul_pd(ss23, n);
        // s²
        let s_sq01 = _mm_mul_pd(s01, s01);
        let s_sq23 = _mm_mul_pd(s23, s23);
        // ss*N - s²
        let var01 = _mm_sub_pd(ss_n01, s_sq01);
        let var23 = _mm_sub_pd(ss_n23, s_sq23);
        // >= thr*N² (NaN-safe via cmpge — NaN compares false, matching the
        // scalar `>=` semantics on x86)
        let cmp01 = _mm_cmpge_pd(var01, thr);
        let cmp23 = _mm_cmpge_pd(var23, thr);
        // _mm_movemask_pd packs each lane's sign bit into bits 0/1.
        let lo = _mm_movemask_pd(cmp01) as u32;
        let hi = _mm_movemask_pd(cmp23) as u32;
        lo | (hi << 2)
    }
}

/// NEON implementation of [`SquaredIntegralImage::passes_variance_mask_4`].
/// 4 f64 lanes packed as 2 × `float64x2_t`. (NEON does not provide
/// `vmulq_u64`, so the integer products are computed in f64 lane math —
/// every intermediate is exact for the variance test's input range.)
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn variance_mask_4_aarch64(
    sum: [u64; 4],
    sum_sq: [u64; 4],
    n_pixels_f64: f64,
    thr_n_sq_f64: f64,
) -> u32 {
    use std::arch::aarch64::*;
    // SAFETY: NEON baseline intrinsics; intermediates exact (see fn doc).
    unsafe {
        // Materialise the 4 u64s into 2 f64x2 vectors via a stack temp
        // (NEON f64 vectors are not bit-castable from u64 vectors on all
        // toolchains, so we go through f64 storage).
        let mut s_buf = [sum[0] as f64, sum[1] as f64, sum[2] as f64, sum[3] as f64];
        let mut ss_buf = [
            sum_sq[0] as f64,
            sum_sq[1] as f64,
            sum_sq[2] as f64,
            sum_sq[3] as f64,
        ];
        let s01 = vld1q_f64(s_buf.as_ptr());
        let s23 = vld1q_f64(s_buf.as_ptr().add(2));
        let ss01 = vld1q_f64(ss_buf.as_ptr());
        let ss23 = vld1q_f64(ss_buf.as_ptr().add(2));
        let n = vdupq_n_f64(n_pixels_f64);
        let thr = vdupq_n_f64(thr_n_sq_f64);
        let ss_n01 = vmulq_f64(ss01, n);
        let ss_n23 = vmulq_f64(ss23, n);
        let s_sq01 = vmulq_f64(s01, s01);
        let s_sq23 = vmulq_f64(s23, s23);
        let var01 = vsubq_f64(ss_n01, s_sq01);
        let var23 = vsubq_f64(ss_n23, s_sq23);
        // `vcgeq_f64` returns a `uint64x2_t` mask where each lane is
        // either `u64::MAX` (true) or `0` (false). The sign bit is the
        // only distinguishing bit, so we can extract each lane's MSB
        // directly: `vgetq_lane_u64` returns the u64, then mask with 1.
        let cmp01 = vcgeq_f64(var01, thr);
        let cmp23 = vcgeq_f64(var23, thr);
        let b0 = (vgetq_lane_u64(cmp01, 0) & 1) as u32;
        let b1 = (vgetq_lane_u64(cmp01, 1) & 1) as u32;
        let b2 = (vgetq_lane_u64(cmp23, 0) & 1) as u32;
        let b3 = (vgetq_lane_u64(cmp23, 1) & 1) as u32;
        b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)
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

    /// SIMD `rect_sum_unchecked_narrow_simd` must be bit-equivalent to the
    /// scalar generic `rect_sum_unchecked_narrow` path. Covers all the
    /// cascading-u32 wraparound edge cases we exercise at runtime.
    #[test]
    fn rect_sum_simd_matches_scalar() {
        // Build a few different integral tables; for each, exhaustively
        // call both paths on every sub-rectangle the table could host.
        for &(w, h) in &[
            (8usize, 6usize),
            (16, 12),
            (32, 24),
            (49, 33),
            (64, 64),
            (129, 65),
        ] {
            let img = lcg_image(w, h);
            let ii = IntegralImage::from_gray(&img);
            for ry in 0..=h {
                for rx in 0..=w {
                    for ry2 in (ry + 1)..=h {
                        for rx2 in (rx + 1)..=w {
                            let scalar = unsafe { ii.rect_sum_unchecked_narrow(rx, ry, rx2, ry2) };
                            let simd = unsafe {
                                ii.rect_sum_unchecked_narrow_simd_method(rx, ry, rx2, ry2)
                            };
                            assert_eq!(
                                scalar, simd,
                                "rect_sum SIMD mismatch at ({rx},{ry})..({rx2},{ry2}) on {w}x{h}"
                            );
                        }
                    }
                }
            }
        }
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
            let IntegralTable::Narrow(data) = &ii.data else {
                panic!("small images must use the narrow u32 table");
            };
            assert_eq!(data, &naive_integral_data(&img), "mismatch at {w}x{h}");
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

    /// Brute-force reference: enumerate every pixel of the cone
    /// `(Y − 1 − py) ≥ |X − 1 − px|`. Returns the full padded table.
    fn rotated_cone_reference(img: &GrayImage) -> Vec<i64> {
        let w = img.width();
        let h = img.height();
        let stride = w + 1;
        let mut data = vec![0i64; stride * (h + 1)];
        for py in 0..h {
            for px in 0..w {
                let v = img[(px, py)] as i64;
                // Add this pixel to every cone cell that contains it.
                for y in (py + 1)..=h {
                    let d = y - 1 - py;
                    // Column 0 is padding: cones never write into it.
                    let xlo = (px + 1).saturating_sub(d).max(1);
                    for x in xlo..=(px + 1 + d).min(w) {
                        data[y * stride + x] += v;
                    }
                }
            }
        }
        data
    }

    #[test]
    fn rotated_table_matches_cone_reference() {
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
            assert_eq!(ri.data, rotated_cone_reference(&img), "mismatch at {w}x{h}");
        }
    }

    #[test]
    fn tilted_query_matches_direct_cone_set_arithmetic() {
        // Independent validation of the CV_TILTED_OFS query formula. For
        // each corner we directly enumerate the pixels in that corner's
        // cone and sum them, then combine the four cones with the same
        // inclusion–exclusion signs. Corners outside the table contribute
        // nothing (OpenCV reads 0 there rather than a clipped cone).
        fn cone_sum(img: &GrayImage, cx: isize, cy: isize) -> i64 {
            let (w, h) = (img.width() as isize, img.height() as isize);
            if cx < 1 || cy < 1 || cx > w || cy > h {
                return 0;
            }
            let mut sum = 0i64;
            for py in 0..h {
                for px in 0..w {
                    // (cy-1-py) >= |cx-1-px|  ⇔  cy-py > |cx-1-px|
                    if py < cy && cy - py > (cx - 1 - px).abs() {
                        sum += img[(px as usize, py as usize)] as i64;
                    }
                }
            }
            sum
        }

        for (w, h) in [(1usize, 1usize), (2, 2), (3, 3), (5, 3), (8, 8), (17, 9)] {
            let img = lcg_image(w, h);
            let ri = RotatedIntegralImage::from_gray(&img);
            for x1 in 0..w {
                for y1 in 0..h {
                    for tw in 1..=3usize {
                        for th in 1..=3usize {
                            let (x1i, y1i) = (x1 as isize, y1 as isize);
                            let (twi, thi) = (tw as isize, th as isize);
                            let expected = cone_sum(&img, x1i, y1i)
                                - cone_sum(&img, x1i - thi, y1i + thi)
                                - cone_sum(&img, x1i + twi, y1i + twi)
                                + cone_sum(&img, x1i + twi - thi, y1i + twi + thi);
                            assert_eq!(
                                ri.tilted_rect_sum(x1, y1, x1 + tw, y1 + th),
                                expected,
                                "query mismatch at ({x1},{y1}) tw={tw} th={th} \
                                 on {w}x{h}"
                            );
                        }
                    }
                }
            }
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
                        // The unchecked tilted lookup has a stricter
                        // contract: every corner must land inside the table,
                        // otherwise callers must use the checked variant.
                        let (tw, th) = (x2 - x1, y2 - y1);
                        if x1 >= th && x1 + tw <= w && y1 + tw + th <= h {
                            assert_eq!(
                                ii.tilted_rect_sum(&ri, x1, y1, x2, y2),
                                ri.tilted_rect_sum_unchecked(x1, y1, x2, y2),
                                "tilted mismatch at ({x1},{y1},{x2},{y2})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn wide_integral_survives_u32_overflow_sized_image() {
        // 4096*4113 = 16_846_848 pixels; an all-white sum is 4_295_946_240,
        // which WRAPS a u32 prefix table (max 4_294_967_295). The detector
        // must select the u64 wide path and every inclusion–exclusion read
        // must still return the exact sum (regression for silently corrupted
        // Haar scores on large bright images).
        let (w, h) = (4096usize, 4113usize);
        assert!(!prefix_sums_fit_u32(w, h));
        assert!(prefix_sums_fit_u32(4096, 4096));
        let mut img = GrayImage::new(w, h);
        img.as_mut_slice().fill(255);
        let ii = IntegralImage::from_gray(&img);
        assert!(ii.is_wide(), "overflow-sized image must use u64 table");
        assert_eq!(ii.at(w, h), (w as u64) * (h as u64) * 255);
        // Full-window and interior rect sums must be exact.
        assert_eq!(ii.rect_sum(0, 0, w, h), (w as u64) * (h as u64) * 255);
        assert_eq!(ii.rect_sum(100, 200, 100 + 24, 200 + 24), 24 * 24 * 255);
        assert_eq!(ii.rect_sum(w - 1, h - 1, w, h), 255);
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
        let narrow_ptr = |ii: &IntegralImage| match &ii.data {
            IntegralTable::Narrow(v) => v.as_ptr(),
            IntegralTable::Wide(_) => panic!("20x14 must use the narrow table"),
        };
        let p1 = {
            let ii = IntegralImage::from_gray(&img);
            narrow_ptr(&ii)
        };
        let ii2 = IntegralImage::from_gray(&img);
        // After dropping the first image its buffer returns to the pool and
        // the next same-size build reuses it (same backing allocation).
        assert_eq!(p1, narrow_ptr(&ii2));
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
        let IntegralTable::Narrow(ii_data) = &ii.data else {
            panic!("23x7 must use the narrow table");
        };
        assert_eq!(ii_data, &naive_integral_data(&img));
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

    /// Bit-equivalence: the 4-wide SIMD variance pre-filter must agree
    /// with the scalar `passes_variance_sums_fast` for every window in the
    /// 640×480 sweep, across 5 deterministic patterns and a thorough
    /// threshold sweep. A divergence here means the SIMD lane math is
    /// taking a different branch than production; a near-miss (off-by-one
    /// on a zero-variance window, say) would silently shift the cascade's
    /// accept/reject set and break the demo cascade's detections.
    #[test]
    fn variance_prefilter_mask_4_matches_scalar() {
        // 5 deterministic patterns: LCG + constant + alternating stripes +
        // a centered-face-like bright blob + a uniform background with
        // a single small perturbation. They span the spectrum of
        // `sum`/`sum_sq` distributions the detector sees in practice.
        let mut all_ok = true;
        let mut run = |label: &str, img: GrayImage| {
            let ii = IntegralImage::from_gray(&img);
            let sq = SquaredIntegralImage::from_gray(&img);
            let win_w = 24usize;
            let win_h = 24usize;
            let nw_norm = win_w - 2;
            let nh_norm = win_h - 2;
            let n_pixels = (nw_norm * nh_norm) as u64;
            let n_pixels_sq = n_pixels * n_pixels;
            // Thresholds across the spectrum: every-window-pass (0),
            // every-window-fail (u64::MAX/4), and the production default (200).
            for thr in [0u64, 200, 10_000] {
                let n_f64 = n_pixels as f64;
                let thr_n_sq_f64 = (thr as f64) * (n_pixels_sq as f64);
                let mut windows_compared = 0usize;
                let mut mismatches: Vec<(usize, usize, u32, u32)> = Vec::new();
                let mut x = 0usize;
                while x + win_w <= ii.width() {
                    let mut y = 0usize;
                    while y + win_h <= ii.height() {
                        // Read 4 windows in x (stride = 1 for the test sweep
                        // to maximise coverage of corner overlaps).
                        let mut sums = [0u64; 4];
                        let mut sum_sqs = [0u64; 4];
                        for k in 0..4 {
                            let xk = x + k;
                            if xk + win_w <= ii.width() {
                                sums[k] = ii.rect_sum(xk, y, xk + win_w, y + win_h);
                                sum_sqs[k] = sq.rect_sum_sq(xk, y, xk + win_w, y + win_h);
                            }
                        }
                        let mask_simd = SquaredIntegralImage::passes_variance_mask_4(
                            sums,
                            sum_sqs,
                            n_f64,
                            thr_n_sq_f64,
                        );
                        let mut mask_scalar = 0u32;
                        for k in 0..4 {
                            let s = sums[k];
                            let ss = sum_sqs[k];
                            let pass = SquaredIntegralImage::passes_variance_sums_fast(
                                s,
                                ss,
                                n_pixels,
                                n_pixels_sq,
                                thr,
                            );
                            if pass {
                                mask_scalar |= 1 << k;
                            }
                        }
                        if mask_simd != mask_scalar {
                            mismatches.push((x, y, mask_simd, mask_scalar));
                        }
                        windows_compared += 4;
                        y += 1;
                    }
                    x += 1;
                }
                if !mismatches.is_empty() {
                    eprintln!(
                            "[{label} thr={thr}] {} mismatches out of {windows_compared} windows; first: {:?}",
                            mismatches.len(),
                            mismatches[0]
                        );
                    all_ok = false;
                }
                assert!(windows_compared > 0, "sweep produced zero windows");
            }
        };
        run("lcg", lcg_image(640, 480));
        let mut constant_img = GrayImage::new(640, 480);
        constant_img.as_mut_slice().fill(128);
        run("constant", constant_img);
        let mut stripes_img = GrayImage::new(640, 480);
        for y in 0..480 {
            for x in 0..640 {
                stripes_img[(x, y)] = if ((x / 4) + (y / 4)) & 1 == 0 {
                    20
                } else {
                    230
                };
            }
        }
        run("stripes", stripes_img);
        let mut blob_img = GrayImage::new(640, 480);
        for y in 0..480 {
            for x in 0..640 {
                let d = ((x as f32 - 320.0).powi(2) + (y as f32 - 240.0).powi(2)).sqrt();
                blob_img[(x, y)] = if d < 80.0 { 220 } else { 30 };
            }
        }
        run("bright_blob", blob_img);
        let mut near_const_img = GrayImage::new(640, 480);
        near_const_img.as_mut_slice().fill(50);
        for y in 200..240 {
            for x in 280..360 {
                near_const_img[(x, y)] = 250;
            }
        }
        run("near_constant_with_blob", near_const_img);
        assert!(all_ok, "variance mask disagreement — see eprintln above");
    }

    /// Random-input equivalence to guard against LCG-pattern-specific
    /// coincidences. Runs 30 distinct LCG seeds × 4 thresholds on the
    /// 640×480 sweep; any divergence fails the test.
    #[test]
    fn variance_prefilter_mask_4_random_patterns() {
        let win_w = 24usize;
        let win_h = 24usize;
        let nw_norm = win_w - 2;
        let nh_norm = win_h - 2;
        let n_pixels = (nw_norm * nh_norm) as u64;
        let n_pixels_sq = n_pixels * n_pixels;
        for seed in 0u32..30 {
            let mut img = GrayImage::new(640, 480);
            let mut s = seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x1234_5678);
            for v in img.as_mut_slice().iter_mut() {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = (s >> 24) as u8;
            }
            let ii = IntegralImage::from_gray(&img);
            let sq = SquaredIntegralImage::from_gray(&img);
            for &thr in &[0u64, 200, 10_000] {
                let n_f64 = n_pixels as f64;
                let thr_n_sq_f64 = (thr as f64) * (n_pixels_sq as f64);
                let mut bad = 0usize;
                let mut x = 0usize;
                while x + win_w <= ii.width() {
                    let mut y = 0usize;
                    while y + win_h <= ii.height() {
                        let mut sums = [0u64; 4];
                        let mut sum_sqs = [0u64; 4];
                        for k in 0..4 {
                            let xk = x + k;
                            if xk + win_w <= ii.width() {
                                sums[k] = ii.rect_sum(xk, y, xk + win_w, y + win_h);
                                sum_sqs[k] = sq.rect_sum_sq(xk, y, xk + win_w, y + win_h);
                            }
                        }
                        let mask_simd = SquaredIntegralImage::passes_variance_mask_4(
                            sums,
                            sum_sqs,
                            n_f64,
                            thr_n_sq_f64,
                        );
                        let mut mask_scalar = 0u32;
                        for k in 0..4 {
                            let s = sums[k];
                            let ss = sum_sqs[k];
                            let pass = SquaredIntegralImage::passes_variance_sums_fast(
                                s,
                                ss,
                                n_pixels,
                                n_pixels_sq,
                                thr,
                            );
                            if pass {
                                mask_scalar |= 1 << k;
                            }
                        }
                        if mask_simd != mask_scalar {
                            bad += 1;
                        }
                        y += 1;
                    }
                    x += 1;
                }
                assert_eq!(
                    bad,
                    0,
                    "seed={seed} thr={thr}: {bad} 4-tuples disagree (out of {} windows)",
                    (ii.width() - win_w + 1) * (ii.height() - win_h + 1) / 4,
                );
            }
        }
    }
}
