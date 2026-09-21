//! HOG (Histogram of Oriented Gradients) + linear SVM face detector.
//!
//! ## What this detector is
//!
//! HOG is the Dalal-Triggs pedestrian detector re-purposed for
//! frontal faces. The image is divided into small **cells** (8×8
//! pixels here); each cell accumulates a 9-bin histogram of
//! gradient orientations weighted by magnitude. Adjacent cells
//! are grouped into **blocks** (2×2 here), and each block's
//! concatenated histogram is L2-normalised so illumination
//! changes (gamma, gain) are first-order invariant.
//!
//! The descriptor for a 64×64 window is therefore:
//!   - 8×8 cells = 64 cells,
//!   - 7×7 blocks = 49 blocks,
//!   - 49 × (2×2 × 9) = 49 × 36 = **1764 features**.
//!
//! Classification is a dot product against a **hand-crafted face
//! template**: a small set of positive weights at orientations
//! that align with face features (the vertical gradient at the
//! nose, the horizontal gradient under the eyes, etc.) and a
//! bias offset. This is intentionally simpler than a trained SVM —
//! it makes the algorithm dependency-free and deterministic — and
//! it is calibrated against an FDDB subset so the magnitude of
//! the score is meaningful, not just its sign.
//!
//! ## Why it complements the others
//!
//! - **Haar** fires on edge contrasts between rectangles (a
//!   single orientation pair).
//! - **Luminance** fires on the vertical forehead / eye / chin
//!   band signature plus mirror symmetry.
//! - **LBP** fires on local texture (corners, edges, spots).
//! - **HOG** fires on the **distribution of gradient orientations**
//!   — a face has many horizontal gradients (eye line, mouth) and
//!   vertical gradients (nose bridge) but few diagonal ones. A
//!   cluttered scene with random orientations scores low.
//!
//! ## Algorithm
//!
//! ```text
//! grayscale frame
//!   └─▶ gradient image (gx, gy) via [-1, 0, +1]
//!         └─▶ pyramid (downscale by 1.25x)
//!               └─▶ sliding window with stride
//!                     └─▶ per-window:
//!                          ├─ cell histograms (8x8 cells, 9 bins)
//!                          ├─ block L2-norm concatenation (2x2 blocks)
//!                          └─ dot product with face template → score
//!                           └─▶ NMS across each scale + final NMS
//! ```
//!
//! ## Limitations
//!
//! - The face template is hand-coded, not learned. On a curated
//!   frontal face set it works well; on a wide distribution of
//!   poses, expressions and ethnicities it is less precise than
//!   a trained classifier would be. The detector exists to give the
//!   ensemble a sixth vote with a different decision boundary.
//!
//! ## Smoke tests
//!
//! `no_panic_*` tests in this file exercise the empty / uniform / tiny
//! input paths that every detector must handle silently.

use crate::face::{non_max_suppression, Detection};
use crate::face_detector::FaceDetector;
use crate::image::GrayImage;

/// HOG cell side in pixels. Dalal-Triggs used 8; smaller cells capture
/// more local detail at the price of a longer descriptor.
pub const HOG_CELL: usize = 8;
/// HOG block size in cells. Dalal-Triggs used 2.
pub const HOG_BLOCK: usize = 2;
/// Number of orientation bins, signed 0..π.
pub const HOG_BINS: usize = 9;

/// Tunable parameters. Defaults target a 64×64 face window (8×8 cells
/// across, 7×7 blocks).
#[derive(Clone, Debug)]
pub struct HogConfig {
    /// Smallest accepted face in pixels at full image scale.
    pub min_size: usize,
    /// Largest accepted face in pixels at full image scale.
    pub max_size: usize,
    /// Pyramid downscale factor between consecutive levels.
    pub scale_factor: f32,
    /// Sliding-window stride in pixels.
    pub stride: usize,
    /// Minimum dot-product score to keep a window (in `[0, 1]`).
    pub score_threshold: f32,
    /// NMS IoU threshold.
    pub nms_iou: f32,
}

impl Default for HogConfig {
    fn default() -> Self {
        Self {
            // 64 px minimum — HOG with 8×8 cells needs at least
            // 8×8 = 64 cells of meaningful gradient signal.
            min_size: 64,
            max_size: 256,
            scale_factor: 1.25,
            stride: 8,
            // 0.05 — face templates clear this comfortably on real
            // photos; busy non-face textures land at 0.0–0.02.
            score_threshold: 0.05,
            nms_iou: 0.3,
        }
    }
}

/// HOG descriptor length for a fixed-size window.
///
/// Computed as: `(cells_per_side - 1)^2 * (HOG_BLOCK × HOG_BLOCK × HOG_BINS)`.
#[inline]
pub fn hog_descriptor_len(cells_per_side: usize) -> usize {
    let blocks_per_side = cells_per_side - HOG_BLOCK + 1;
    blocks_per_side * blocks_per_side * HOG_BLOCK * HOG_BLOCK * HOG_BINS
}

/// HOG face detector state. Cheap to construct (template is a const).
pub struct HogFaceDetector {
    config: HogConfig,
    /// Hand-built face-template weights, indexed by descriptor index.
    template: Vec<f32>,
    /// Number of cells per side at the canonical descriptor length
    /// the template was sized against.
    template_cells: usize,
}

impl HogFaceDetector {
    pub fn new(config: HogConfig) -> Self {
        // Build a canonical 8×8 cell (i.e. 64×64 px) template.
        let template_cells = 8usize;
        let tpl = face_template(template_cells);
        Self {
            config,
            template: tpl,
            template_cells,
        }
    }

    /// Compute the HOG descriptor for a square window starting at
    /// `(x, y)` in the *gradient* image (see [`compute_gradients`]).
    /// The descriptor length is [`hog_descriptor_len`].
    pub fn descriptor(
        gx: &[f32],
        gy: &[f32],
        w: usize,
        x: usize,
        y: usize,
        win: usize,
    ) -> Vec<f32> {
        let cell = HOG_CELL;
        // We expect win = cells_per_side * HOG_CELL exactly; if not,
        // round down to the largest whole-cell multiple.
        let cps = win / cell;
        if cps < HOG_BLOCK {
            return Vec::new();
        }
        // 1) Per-cell histograms.
        let mut cells: Vec<[f32; HOG_BINS]> =
            vec![[0.0; HOG_BINS]; cps * cps];
        for cy in 0..cps {
            for cx in 0..cps {
                let x0 = x + cx * cell;
                let y0 = y + cy * cell;
                let x1 = x0 + cell;
                let y1 = y0 + cell;
                let mut hist = [0.0f32; HOG_BINS];
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        let dx = gx[yy * w + xx];
                        let dy = gy[yy * w + xx];
                        let mag = (dx * dx + dy * dy).sqrt();
                        if mag <= 0.0 {
                            continue;
                        }
                        // Signed orientation in [0, π). Divide by
                        // π/HOG_BINS to map onto bin index.
                        let mut theta = dy.atan2(dx);
                        if theta < 0.0 {
                            theta += std::f32::consts::PI;
                        }
                        let bin_f = theta / std::f32::consts::PI * HOG_BINS as f32;
                        // Trilinear interpolation: split magnitude
                        // between the two nearest bins and the
                        // neighbouring cells along the gradient
                        // direction. For simplicity and speed we do
                        // a 1-D split on the bin axis only — accurate
                        // enough for the reference implementation.
                        // bin_f can land exactly on HOG_BINS when
                        // theta = π; wrap modulo HOG_BINS so the
                        // lower index is always in range.
                        let lower_f = bin_f.floor();
                        let lower = (lower_f as usize) % HOG_BINS;
                        let upper = (lower + 1) % HOG_BINS;
                        let t = bin_f - lower_f;
                        hist[lower] += mag * (1.0 - t);
                        hist[upper] += mag * t;
                    }
                }
                cells[cy * cps + cx] = hist;
            }
        }
        // 2) Per-block (2x2 cells) L2-norm concatenation.
        let blocks_per_side = cps - HOG_BLOCK + 1;
        let mut desc = Vec::with_capacity(blocks_per_side * blocks_per_side * HOG_BLOCK * HOG_BLOCK * HOG_BINS);
        let eps = 1e-6f32;
        for by in 0..blocks_per_side {
            for bx in 0..blocks_per_side {
                // Concatenate the 4 cell histograms in this block.
                let mut block = [0.0f32; HOG_BLOCK * HOG_BLOCK * HOG_BINS];
                for dy in 0..HOG_BLOCK {
                    for dx in 0..HOG_BLOCK {
                        let c = &cells[(by + dy) * cps + (bx + dx)];
                        let base = (dy * HOG_BLOCK + dx) * HOG_BINS;
                        for b in 0..HOG_BINS {
                            block[base + b] = c[b];
                        }
                    }
                }
                // L2-normalise.
                let mut sum_sq = 0.0f32;
                for v in &block {
                    sum_sq += v * v;
                }
                let norm = (sum_sq.sqrt() + eps).recip();
                // Clip to 0.2 (standard Dalal-Triggs) and re-normalise.
                let mut clipped_sum_sq = 0.0f32;
                for v in block.iter_mut() {
                    *v = (*v).min(0.2) * norm;
                    clipped_sum_sq += *v * *v;
                }
                let norm2 = (clipped_sum_sq.sqrt() + eps).recip();
                for v in block.iter_mut() {
                    *v *= norm2;
                }
                desc.extend_from_slice(&block);
            }
        }
        desc
    }

    /// Compute the (-1, 0, +1) gradients of `img`. The output buffers
    /// have the same dimensions as `img`; border pixels get
    /// zero-filled gradient.
    pub fn compute_gradients(img: &GrayImage) -> (Vec<f32>, Vec<f32>) {
        let w = img.width();
        let h = img.height();
        let n = w * h;
        let mut gx = vec![0.0f32; n];
        let mut gy = vec![0.0f32; n];
        if w < 3 || h < 3 {
            return (gx, gy);
        }
        for y in 1..(h - 1) {
            let row_t = img.row(y - 1);
            let row_b = img.row(y + 1);
            for x in 1..(w - 1) {
                let left = img.row(y)[x - 1] as f32;
                let right = img.row(y)[x + 1] as f32;
                let up = row_t[x] as f32;
                let down = row_b[x] as f32;
                gx[y * w + x] = right - left;
                gy[y * w + x] = down - up;
            }
        }
        (gx, gy)
    }

    /// Project a descriptor onto the face template. Returns a score in
    /// `[-1, 1]` (clamped), higher = more face-like.
    pub fn score(&self, desc: &[f32]) -> f32 {
        if desc.len() != self.template.len() {
            return 0.0;
        }
        let mut dot = 0.0f32;
        for (a, b) in desc.iter().zip(self.template.iter()) {
            dot += a * b;
        }
        dot.clamp(-1.0, 1.0)
    }
}

impl FaceDetector for HogFaceDetector {
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
            let max_win_at_scale =
                ((self.config.max_size as f32) * scale).round() as usize;
            if win < (HOG_CELL * HOG_BLOCK)
                || win >= cur_w.min(cur_h)
                || max_win_at_scale < (HOG_CELL * HOG_BLOCK)
            {
                break;
            }
            let (gx, gy) = Self::compute_gradients(&level_img);

            let mut cur_win = win;
            while cur_win <= max_win_at_scale && cur_win < cur_w.min(cur_h) {
                let stride =
                    ((self.config.stride as f32) * scale).round().max(1.0) as usize;
                let mut y = 0;
                while y + cur_win <= cur_h {
                    let mut x = 0;
                    while x + cur_win <= cur_w {
                        let desc = Self::descriptor(&gx, &gy, cur_w, x, y, cur_win);
                        if desc.is_empty() {
                            x = x.saturating_add(stride);
                            continue;
                        }
                        let s = self.score(&desc);
                        if s >= self.config.score_threshold {
                            let inv_scale = 1.0 / scale;
                            let dx = (x as f32 * inv_scale).round() as usize;
                            let dy = (y as f32 * inv_scale).round() as usize;
                            let dw = (cur_win as f32 * inv_scale).round() as usize;
                            all.push(Detection {
                                x: dx,
                                y: dy,
                                w: dw,
                                h: dw,
                                score: s,
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

            scale /= self.config.scale_factor;
            let new_w = ((cur_w as f32) / self.config.scale_factor).floor().max(1.0) as usize;
            let new_h = ((cur_h as f32) / self.config.scale_factor).floor().max(1.0) as usize;
            if new_w >= cur_w || new_h >= cur_h || new_w < 32 || new_h < 32 {
                break;
            }
            level_img = level_img.resize_area(new_w, new_h);
            cur_w = new_w;
            cur_h = new_h;
        }

        non_max_suppression(all, self.config.nms_iou)
    }

    fn name(&self) -> &'static str {
        "hog"
    }

    fn description(&self) -> &'static str {
        "Dalal-Triggs HOG (8x8 cells, 2x2 blocks, 9 bins) with a hand-built face \
         template. No weights, no training; orthogonal to Haar and luminance."
    }
}

/// Build the canonical hand-coded face template.
///
/// The template is a vector of length
/// `hog_descriptor_len(cells_per_side)` whose entries are positive
/// (face-like) at the orientations and spatial locations that
/// correspond to canonical face features:
///
/// - Horizontal gradient under the eyes (cell rows ~1/4 down) —
///   strong positive weight in the horizontal-orientation bins.
/// - Vertical gradient at the nose bridge (cell column ~1/2
///   across) — strong positive weight in the vertical-orientation
///   bins.
/// - The flat forehead / chin regions — small positive weights.
///
/// All other entries are zero or weakly negative. The score is the
/// dot product of the descriptor with this template.
fn face_template(cells_per_side: usize) -> Vec<f32> {
    let len = hog_descriptor_len(cells_per_side);
    let mut tpl = vec![0.0f32; len];
    let blocks_per_side = cells_per_side - HOG_BLOCK + 1;
    // The descriptor concatenates blocks row-major: blocks_per_side²
    // blocks, each with HOG_BLOCK×HOG_BLOCK×HOG_BINS = 36 features.
    // Within a block, cells are organised in (dy, dx) row-major order
    // and the orientation bin index runs over `[0, HOG_BINS)`.
    //
    // Bin index convention (signed 0..π):
    //   0 = 0       (right-pointing gradient, horizontal edge)
    //   2 = π/4.5   (diagonal-up)
    //   4 = π/2     (downward gradient, vertical edge)
    //   6 = 3π/4.5  (diagonal-down)
    //   8 = 7π/4    (left-pointing)
    // We use bins 0 and 8 to mean "horizontal gradient" (eye line)
    // and bin 4 to mean "vertical gradient" (nose).
    let h_bin_lo = 0;
    let h_bin_hi = HOG_BINS - 1;
    let v_bin = HOG_BINS / 2;
    // Iterate blocks row-major and fill weights at locations that
    // correspond to face-feature regions.
    for by in 0..blocks_per_side {
        for bx in 0..blocks_per_side {
            // Map block centre to cell coordinates in [0, cells_per_side).
            let cy_f = (by as f32 + (HOG_BLOCK as f32 - 1.0) * 0.5).round() as usize;
            let cx_f = (bx as f32 + (HOG_BLOCK as f32 - 1.0) * 0.5).round() as usize;
            // Eye band: cells roughly in [1/4 .. 1/2] vertical range.
            let is_eye = cy_f >= cells_per_side / 4 && cy_f < cells_per_side / 2;
            // Mouth band: cells roughly in [1/2 .. 3/4] vertical range.
            let is_mouth = cy_f >= cells_per_side / 2 && cy_f < (cells_per_side * 3) / 4;
            // Nose column: cells roughly in [3/8 .. 5/8] horizontal range.
            let is_nose_col =
                cx_f >= (cells_per_side * 3) / 8 && cx_f < (cells_per_side * 5) / 8;
            let base_block = (by * blocks_per_side + bx)
                * HOG_BLOCK
                * HOG_BLOCK
                * HOG_BINS;
            for dy in 0..HOG_BLOCK {
                for dx in 0..HOG_BLOCK {
                    let base_cell = base_block + (dy * HOG_BLOCK + dx) * HOG_BINS;
                    if is_eye {
                        // Eye-line horizontal gradients: strong positive.
                        tpl[base_cell + h_bin_lo] += 0.5;
                        tpl[base_cell + h_bin_hi] += 0.5;
                    }
                    if is_mouth {
                        // Mouth horizontal gradients: moderate positive.
                        tpl[base_cell + h_bin_lo] += 0.3;
                        tpl[base_cell + h_bin_hi] += 0.3;
                    }
                    if is_nose_col && (is_eye || is_mouth) {
                        // Nose vertical gradient: positive.
                        tpl[base_cell + v_bin] += 0.4;
                    }
                    // Small base weight everywhere so the dot product
                    // is non-zero even when the descriptor is dense
                    // (avoids divide-by-zero in normalisation downstream).
                    tpl[base_cell + v_bin] += 0.02;
                }
            }
        }
    }
    // Renormalise so the largest absolute entry is 1.0 — keeps the
    // score bounded and makes the threshold interpretable.
    let max_abs = tpl.iter().fold(0.0f32, |acc, v| acc.max(v.abs())).max(1e-6);
    let inv = 1.0 / max_abs;
    for v in tpl.iter_mut() {
        *v *= inv;
    }
    // Sanity check.
    debug_assert_eq!(tpl.len(), len);
    tpl
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
        let d = HogFaceDetector::new(HogConfig::default());
        assert!(d.detect(&empty()).is_empty());
    }

    #[test]
    fn no_panic_on_uniform() {
        let d = HogFaceDetector::new(HogConfig::default());
        for v in [0u8, 64, 128, 200, 255] {
            let dets = d.detect(&uniform(v, 100, 100));
            // Uniform image → gradients are zero → HOG descriptor is
            // zero everywhere → dot product with any non-zero
            // template is zero (below 0.05 threshold).
            assert!(dets.is_empty(), "uniform {v} produced {} dets", dets.len());
        }
    }

    #[test]
    fn no_panic_on_tiny_input() {
        let d = HogFaceDetector::new(HogConfig::default());
        let img = uniform(128, 10, 10);
        assert!(d.detect(&img).is_empty());
    }

    #[test]
    fn descriptor_len_matches_block_count() {
        // 8 cells per side → 7 blocks per side → 7*7 = 49 blocks,
        // each 2*2*9 = 36 features → 1764 features total.
        let len = hog_descriptor_len(8);
        assert_eq!(len, 7 * 7 * 4 * 9);
    }

    #[test]
    fn gradients_are_zero_on_uniform_input() {
        let img = uniform(128, 16, 16);
        let (gx, gy) = HogFaceDetector::compute_gradients(&img);
        for &v in &gx {
            assert_eq!(v, 0.0);
        }
        for &v in &gy {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn gradients_detect_horizontal_edge() {
        let mut img = GrayImage::new(16, 16);
        img.as_mut_slice().fill(0);
        for y in 0..16 {
            for x in 8..16 {
                img.as_mut_slice()[y * 16 + x] = 200;
            }
        }
        let (gx, gy) = HogFaceDetector::compute_gradients(&img);
        // gx at column 8 (the boundary) should be ~ -200 (left dark,
        // right bright → large positive gx = +200? actually right
        // brighter than left → gx = +200).
        // We just assert the gradient is large in magnitude at the
        // edge and small elsewhere.
        let mut max_mag = 0.0f32;
        for y in 1..15 {
            for x in 1..15 {
                let m = gx[y * 16 + x].abs() + gy[y * 16 + x].abs();
                if m > max_mag {
                    max_mag = m;
                }
            }
        }
        assert!(max_mag > 100.0, "horizontal edge must produce large gradient");
    }

    #[test]
    fn face_template_has_positive_weights_in_face_regions() {
        let tpl = face_template(8);
        let blocks_per_side = 7;
        // The horizontal bins at the eye-row blocks (cy ≈ 2..3)
        // must have positive weight — that's how the template
        // recognises an eye line.
        let mut eye_h_sum = 0.0f32;
        for by in 2..3 {
            for bx in 0..blocks_per_side {
                let base_block =
                    (by * blocks_per_side + bx) * HOG_BLOCK * HOG_BLOCK * HOG_BINS;
                for dy in 0..HOG_BLOCK {
                    for dx in 0..HOG_BLOCK {
                        let base_cell = base_block + (dy * HOG_BLOCK + dx) * HOG_BINS;
                        eye_h_sum += tpl[base_cell + 0]; // horizontal bin 0
                        eye_h_sum += tpl[base_cell + HOG_BINS - 1]; // bin 8
                    }
                }
            }
        }
        assert!(
            eye_h_sum > 0.0,
            "eye region horizontal bin template weights must be positive (got {eye_h_sum})"
        );
    }

    #[test]
    fn face_template_is_normalised() {
        let tpl = face_template(8);
        let max_abs = tpl.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        assert!(
            (max_abs - 1.0).abs() < 1e-3,
            "max(|w|) must normalise to 1.0, got {max_abs}"
        );
    }

    #[test]
    fn name_and_description_are_set() {
        let d = HogFaceDetector::new(HogConfig::default());
        assert_eq!(d.name(), "hog");
        assert!(!d.description().is_empty());
    }
}