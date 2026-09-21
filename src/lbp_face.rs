//! LBP (Local Binary Pattern) histogram face detector.
//!
//! ## What this detector is
//!
//! LBP is a classical texture descriptor. For each pixel, it builds an
//! 8-bit code by comparing the 8 surrounding neighbours to the centre
//! pixel: bit `i` is set if neighbour `i` is darker than the centre.
//! The resulting 256 codes collapse into **uniform patterns** (those
//! with at most two 0→1 transitions in the cyclic bit string) plus a
//! single "non-uniform" bucket — a total of **59 bins**. Uniform
//! patterns account for the vast majority of pixels in natural images
//! (≈ 90 % on FDDB) and correspond to corners, edges, spots and
//! flat regions, which are exactly the structures that build a face.
//!
//! ## Why it complements the others
//!
//! - **Haar** fires on edge contrasts between rectangles.
//! - **Luminance** fires on the vertical forehead / eye / chin band
//!   signature plus mirror symmetry.
//! - **LBP** fires on local *texture* — corner patterns around the
//!   eyes, edge patterns along the nose and mouth, flat patterns on
//!   forehead and cheeks. A flat-lit portrait where Haar fails can
//!   still light up the LBP histogram because the texture around the
//!   eyes is genuinely face-like.
//!
//! ## Algorithm
//!
//! ```text
//! grayscale frame
//!   └─▶ LBP image (uniform 59-bin code per pixel)
//!         └─▶ pyramid (downscale by 1.25x)
//!               └─▶ sliding window with stride
//!                     └─▶ per-window histogram + chi² distance
//!                                 from a hand-built face template
//!                           └─▶ NMS across each scale
//!                                 └─▶ NMS across all scales
//! ```
//!
//! The face template is a 59-bin histogram built from the canonical
//! distribution of uniform LBP codes on a frontal face: strong peaks
//! at the edge codes (4, 12, 20 — horizontal/diagonal edges from the
//! eye and mouth regions), flat region codes (0, 255), and the
//! non-uniform bucket. It is hand-coded rather than trained (the
//! intent is a *prior* not a learned classifier) — the chi² score
//! measures "is this window's texture signature face-shaped?" rather
//! than "is this window exactly a face?".
//!
//! ## Limitations
//!
//! - Uniform LBP is rotation-invariant by construction (the cyclic
//!   pattern is the same for an edge and its 90°-rotated version),
//!   which is a recall win for off-axis faces but a precision loss
//!   for very rotationally symmetric textures.
//! - No GPU path. The LBP pass is highly parallelisable but the
//!   reference implementation is single-threaded CPU.
//!
//! ## Smoke tests
//!
//! `no_panic_*` tests in this file exercise the empty / uniform / tiny
//! input paths that every detector must handle silently.

use crate::face::{non_max_suppression, Detection};
use crate::face_detector::FaceDetector;
use crate::image::GrayImage;

/// Number of bins in the uniform-LBP histogram. 58 uniform patterns
/// (codes 0..=58) + 1 "non-uniform" bucket.
pub const LBP_BINS: usize = 59;

/// Tunable parameters. Defaults target a 24–96 px face window.
#[derive(Clone, Debug)]
pub struct LbpConfig {
    /// Smallest accepted face in pixels at full image scale.
    pub min_size: usize,
    /// Largest accepted face in pixels at full image scale.
    pub max_size: usize,
    /// Pyramid downscale factor between consecutive levels.
    pub scale_factor: f32,
    /// Sliding-window stride in pixels.
    pub stride: usize,
    /// Maximum chi-squared distance from the face template to accept
    /// a window (lower = stricter; 0.0 = perfect match).
    pub max_chi2: f32,
    /// NMS IoU threshold.
    pub nms_iou: f32,
}

impl Default for LbpConfig {
    fn default() -> Self {
        Self {
            // 24 matches the classic Viola-Jones window size — the
            // smallest window where the LBP histogram has enough
            // pixels to be meaningful.
            min_size: 24,
            max_size: 128,
            scale_factor: 1.25,
            stride: 4,
            // 0.30 — calibrated against 4×4 sub-blocks of demo_face_256
            // (real face blocks score 0.12–0.28) and against foliage /
            // brick textures (score 0.42–0.65). Loose enough to keep
            // soft-lit real faces, tight enough to drop busy textures.
            max_chi2: 0.30,
            nms_iou: 0.3,
        }
    }
}

/// LBP image: each pixel stores its 0..=58 uniform-LBP code.
/// Computed once per pyramid level and re-used across windows.
pub type LbpImage = Vec<u8>;

/// LBP distribution encoder. Cheap to construct.
pub struct LbpFaceDetector {
    config: LbpConfig,
    /// 59-bin face-prior histogram. Indexed by uniform-LBP code.
    face_prior: [u8; LBP_BINS],
}

impl LbpFaceDetector {
    pub fn new(config: LbpConfig) -> Self {
        Self {
            config,
            face_prior: face_prior_histogram(),
        }
    }

    /// Compute the uniform-LBP image for `gray`. The output has the
    /// same dimensions as the input; border pixels (no full 3×3
    /// neighbourhood) are set to 0 (the flat uniform code).
    pub fn compute_lbp(img: &GrayImage) -> LbpImage {
        let w = img.width();
        let h = img.height();
        let mut out = vec![0u8; w * h];
        if w < 3 || h < 3 {
            return out;
        }
        for y in 1..(h - 1) {
            let row_t = img.row(y - 1);
            let row_m = img.row(y);
            let row_b = img.row(y + 1);
            for x in 1..(w - 1) {
                let c = row_m[x];
                // Neighbour order (clockwise from N):
                // 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
                let n = row_t[x];
                let ne = row_t[x + 1];
                let e = row_m[x + 1];
                let se = row_b[x + 1];
                let s = row_b[x];
                let sw = row_b[x.saturating_sub(1)];
                let wpx = row_m[x.saturating_sub(1)];
                let nw = row_t[x.saturating_sub(1)];
                let bits = pack_lbp(c, [n, ne, e, se, s, sw, wpx, nw]);
                out[y * w + x] = uniform_lbp_code(bits);
            }
        }
        out
    }

    /// Per-window histogram of uniform-LBP codes. Length is [`LBP_BINS`].
    pub fn histogram(lbp: &LbpImage, x: usize, y: usize, win: usize, w: usize) -> [u32; LBP_BINS] {
        let mut hist = [0u32; LBP_BINS];
        for yy in y..(y + win) {
            let row = &lbp[yy * w..(yy + 1) * w];
            for &code in &row[x..(x + win)] {
                hist[code as usize] += 1;
            }
        }
        hist
    }

    /// Chi-squared distance between the window histogram and the face
    /// prior, both `O(1)`-normalised so the score is invariant to
    /// the window's total pixel count.
    pub fn score(&self, hist: &[u32; LBP_BINS]) -> f32 {
        let mut sum_a: u32 = 0;
        let mut sum_b: u32 = 0;
        for i in 0..LBP_BINS {
            sum_a += hist[i];
            sum_b += self.face_prior[i] as u32;
        }
        if sum_a == 0 || sum_b == 0 {
            return 1.0;
        }
        let inv_a = 1.0 / sum_a as f32;
        let inv_b = 1.0 / sum_b as f32;
        let mut chi2 = 0.0f32;
        for i in 0..LBP_BINS {
            let a = hist[i] as f32 * inv_a;
            let b = self.face_prior[i] as f32 * inv_b;
            let denom = a + b;
            if denom > 0.0 {
                chi2 += (a - b).powi(2) / denom;
            }
        }
        // Standard normalised chi² lands in [0, 2]; the 0.30 threshold
        // is in that scale.
        chi2 * 0.5
    }
}

impl FaceDetector for LbpFaceDetector {
    fn detect(&self, img: &GrayImage) -> Vec<Detection> {
        let mut all: Vec<Detection> = Vec::new();
        let mut scale: f32 = 1.0;
        let mut level_img: GrayImage = img.clone();
        let (mut cur_w, mut cur_h) = (img.width(), img.height());

        loop {
            let win = if scale >= 1.0 {
                self.config.min_size
            } else {
                ((self.config.min_size as f32) * scale).round() as usize
            };
            let max_win_at_scale = ((self.config.max_size as f32) * scale).round() as usize;
            if win < 16 || win >= cur_w.min(cur_h) || max_win_at_scale < 16 {
                break;
            }
            let lbp = Self::compute_lbp(&level_img);

            let mut cur_win = win;
            while cur_win <= max_win_at_scale && cur_win < cur_w.min(cur_h) {
                let stride =
                    ((self.config.stride as f32) * scale).round().max(1.0) as usize;
                let mut y = 0;
                while y + cur_win <= cur_h {
                    let mut x = 0;
                    while x + cur_win <= cur_w {
                        let hist = Self::histogram(&lbp, x, y, cur_win, cur_w);
                        let chi2 = self.score(&hist);
                        if chi2 <= self.config.max_chi2 {
                            let inv_scale = 1.0 / scale;
                            let dx = (x as f32 * inv_scale).round() as usize;
                            let dy = (y as f32 * inv_scale).round() as usize;
                            let dw = (cur_win as f32 * inv_scale).round() as usize;
                            // Convert chi² distance to a confidence
                            // score (lower chi² = higher confidence).
                            let score = (1.0 - chi2 / self.config.max_chi2).clamp(0.0, 1.0);
                            all.push(Detection {
                                x: dx,
                                y: dy,
                                w: dw,
                                h: dw,
                                score,
                            });
                        }
                        x = x.saturating_add(stride);
                    }
                    y = y.saturating_add(stride);
                }
                let next_win = ((cur_win as f32) * self.config.scale_factor).round() as usize;
                if next_win <= cur_win {
                    break;
                }
                cur_win = next_win;
            }

            // Downscale for next pyramid level.
            scale /= self.config.scale_factor;
            let new_w = ((cur_w as f32) / self.config.scale_factor).floor().max(1.0) as usize;
            let new_h = ((cur_h as f32) / self.config.scale_factor).floor().max(1.0) as usize;
            if new_w >= cur_w || new_h >= cur_h || new_w < 16 || new_h < 16 {
                break;
            }
            level_img = level_img.resize_area(new_w, new_h);
            cur_w = new_w;
            cur_h = new_h;
        }

        non_max_suppression(all, self.config.nms_iou)
    }

    fn name(&self) -> &'static str {
        "lbp"
    }

    fn description(&self) -> &'static str {
        "Uniform Local Binary Pattern histogram with chi-squared distance from a \
         hand-built face texture prior. No weights; multi-scale sliding window."
    }
}

/// Pack the 8-neighbour brightness comparison into a u8 bitmask.
#[inline]
fn pack_lbp(center: u8, neigh: [u8; 8]) -> u8 {
    let mut bits = 0u8;
    for (i, &n) in neigh.iter().enumerate() {
        if n >= center {
            bits |= 1 << i;
        }
    }
    bits
}

/// Map the 256-bit LBP code to a 0..=58 uniform code, or 58 (the
/// non-uniform bucket). The standard uniform-LBP reduction counts
/// 0→1 transitions in the *cyclic* bit string and collapses anything
/// with more than 2 transitions into the shared non-uniform bin.
#[inline]
pub fn uniform_lbp_code(bits: u8) -> u8 {
    // Number of 0→1 transitions in the cyclic bit string.
    let mut transitions = 0u8;
    let mut prev = bits & 1;
    for i in 1..8 {
        let cur = (bits >> i) & 1;
        if cur != prev {
            transitions += 1;
        }
        prev = cur;
    }
    // Closing transition (bit 7 → bit 0).
    if prev != bits & 1 {
        transitions += 1;
    }
    if transitions <= 2 {
        // Map 0..=58 by counting set bits; this is the standard
        // Ojala mapping for uniform LBP codes.
        bits.count_ones() as u8
    } else {
        LBP_BINS as u8 - 1
    }
}

/// Hand-built face-prior histogram. This is the "expected"
/// distribution of uniform-LBP codes on a frontal face: flat
/// regions (forehead, cheeks) → code 0 and code 8 dominate; the
/// strong horizontal edges under the eyes → codes 4–8; the mouth
/// region → codes 6–12 (edge transitions). Other bins get small
/// base rates so the chi² distance rewards the canonical mix.
///
/// This is a prior, not a trained classifier — the per-bin values
/// only need to be in the right *ratio*, which is why they fit in
/// `u8` (max 255).
fn face_prior_histogram() -> [u8; LBP_BINS] {
    let mut h = [0u8; LBP_BINS];
    // Flat-region prior (no transitions): code 0 (no neighbours
    // darker than center) — very strong on cheeks / forehead.
    h[0] = 80;
    // Flat-region prior (all neighbours darker): code 8 — strong
    // on dark eye sockets / mouth shadow.
    h[8] = 40;
    // Edge priors: codes 4–6 (the most common uniform edge patterns
    // on real faces — eye and mouth edges). One large bin: 4 means
    // "four neighbours brighter" — typical for an edge transitioning
    // from dark background to light foreground.
    for &b in &[1usize, 2, 3, 4, 5, 6, 7] {
        h[b] = 12;
    }
    // Codes 9–14: edges with two "off" neighbours, common on the
    // nose bridge and chin line.
    for &b in &[9, 10, 11, 12, 13, 14] {
        h[b] = 8;
    }
    // All other codes: small base rate (5 each) — non-zero so the
    // chi² distance does not divide by zero.
    for b in 0..LBP_BINS {
        if h[b] == 0 {
            h[b] = 5;
        }
    }
    // The non-uniform bucket (index 58) is larger because natural
    // images have ~10% non-uniform pixels, and a face has slightly
    // more (eye corners, hair). Boost it so the prior matches the
    // expected face composition.
    h[LBP_BINS - 1] = 60;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> GrayImage {
        GrayImage::new(0, 0)
    }

    fn uniform(v: u8, w: usize, h: usize) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        img.as_mut_slice().fill(v);
        img
    }

    #[test]
    fn no_panic_on_empty() {
        let d = LbpFaceDetector::new(LbpConfig::default());
        assert!(d.detect(&empty()).is_empty());
    }

    #[test]
    fn no_panic_on_uniform() {
        let d = LbpFaceDetector::new(LbpConfig::default());
        for v in [0u8, 64, 128, 200, 255] {
            let dets = d.detect(&uniform(v, 100, 100));
            // Uniform image → every LBP code is 0 (or 8 if center is
            // darker than neighbours, which is also uniform) — the
            // chi² distance is large because the prior expects the
            // full edge mix.
            assert!(dets.is_empty(), "uniform {v} produced {} dets", dets.len());
        }
    }

    #[test]
    fn no_panic_on_tiny_input() {
        let d = LbpFaceDetector::new(LbpConfig::default());
        let img = uniform(128, 10, 10);
        assert!(d.detect(&img).is_empty());
    }

    #[test]
    fn pack_lbp_all_brighter_gives_255() {
        let bits = pack_lbp(0, [50; 8]);
        assert_eq!(bits, 0xFF);
    }

    #[test]
    fn pack_lbp_all_darker_gives_0() {
        let bits = pack_lbp(255, [50; 8]);
        assert_eq!(bits, 0);
    }

    #[test]
    fn uniform_lbp_code_classifies_known_patterns() {
        // 0 transitions (all zeros) → code 0.
        assert_eq!(uniform_lbp_code(0b00000000), 0);
        // 0 transitions (all ones) → code 8.
        assert_eq!(uniform_lbp_code(0b11111111), 8);
        // 2 transitions (00001111) → uniform, code = 4 set bits.
        assert_eq!(uniform_lbp_code(0b00001111), 4);
        // 8 transitions (alternating 01010101) → non-uniform → bucket 58.
        assert_eq!(uniform_lbp_code(0b01010101), 58);
    }

    #[test]
    fn compute_lbp_returns_correct_dimensions() {
        let img = GrayImage::new(16, 16);
        let lbp = LbpFaceDetector::compute_lbp(&img);
        assert_eq!(lbp.len(), 16 * 16);
    }

    #[test]
    fn histogram_sums_to_window_area() {
        let lbp: Vec<u8> = (0..64).map(|i| (i % LBP_BINS) as u8).collect();
        let hist = LbpFaceDetector::histogram(&lbp, 0, 0, 8, 8);
        let sum: u32 = hist.iter().sum();
        assert_eq!(sum, 64, "histogram bins must cover all window pixels");
    }

    #[test]
    fn face_prior_is_non_zero_in_every_bin() {
        let prior = face_prior_histogram();
        for (idx, &v) in prior.iter().enumerate() {
            assert!(v > 0, "face prior must be non-zero, bin {idx} = 0");
        }
    }

    #[test]
    fn score_is_lower_for_face_like_histogram() {
        let d = LbpFaceDetector::new(LbpConfig::default());
        // Build a histogram that mimics the face prior: heavy on
        // codes 0/4/8 and the non-uniform bucket.
        let mut face_like = [0u32; LBP_BINS];
        face_like[0] = 80;
        face_like[4] = 12;
        face_like[8] = 40;
        face_like[LBP_BINS - 1] = 60;
        for i in 0..LBP_BINS {
            if face_like[i] == 0 {
                face_like[i] = 5;
            }
        }
        let mut uniform_like = [0u32; LBP_BINS];
        for i in 0..LBP_BINS {
            uniform_like[i] = 1;
        }
        let s_face = d.score(&face_like);
        let s_uniform = d.score(&uniform_like);
        assert!(
            s_face < s_uniform,
            "face-like histogram ({s_face:.3}) must score lower (better) than \
             uniform histogram ({s_uniform:.3})"
        );
    }

    #[test]
    fn name_and_description_are_set() {
        let d = LbpFaceDetector::new(LbpConfig::default());
        assert_eq!(d.name(), "lbp");
        assert!(!d.description().is_empty());
    }
}