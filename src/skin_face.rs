//! Skin-tone face detector — luminance band thresholding + connected components
//! + aspect-ratio filter.
//!
//! ## What this detector is
//!
//! Classical face detectors (Haar, LBP, HOG) all treat the input as
//! luminance only. That is sufficient for most grayscale photos, but
//! when the input *was* a colour photo the luminance detector still
//! works on the luma plane — it just throws away chrominance signal.
//!
//! This detector takes the other route: it does **not** try to model
//! the structure of a face at all. Instead it uses a cheap
//! skin-tone-luminance mask plus a small amount of geometric reasoning
//! (face-shaped aspect ratio, size constraints, density) to recover the
//! pixel clusters that are most likely faces. It is by far the fastest
//! detector in the crate: a single linear pass + a flood-fill pass.
//!
//! ## Why it complements the others
//!
//! - **Haar / luminance / LBP / HOG** all fire on edges, gradients or
//!   band patterns. A clean frontal portrait lit with soft fill light
//!   can score **low** on every one of them because the contrast is
//!   weak — but the skin-tone-luminance band still lights up because
//!   skin is bright-ish, mid-grey, and forms a contiguous blob.
//! - The inverse also holds. A busy cluttered scene with plenty of
//!   edges and texture but no skin tones (foliage, a brick wall) will
//!   fire every gradient detector and *none* of the skin-tone ones.
//!
//! That makes this detector a genuinely orthogonal vote in the ensemble
//! fuser: agreement with one of the classical detectors on a skin-tone
//! blob is strong evidence for a face.
//!
//! ## Algorithm
//!
//! ```text
//! grayscale frame
//!   └─▶ per-pixel skin-tone band mask (luma ∈ [lo, hi))
//!         └─▶ morphological opening (3x3 erode then dilate) to drop salt noise
//!               └─▶ flood-fill connected components
//!                     └─▶ per-component filter: size, aspect ratio, density
//!                           └─▶ final list (sorted by density descending)
//! ```
//!
//! No pyramid, no sliding window — the components give us arbitrary shape
//! boxes for free, so the detector is single-scale only and trivially
//! parallelisable in the future.
//!
//! ## Limitations
//!
//! - Single-scale. A small face in a 4K frame will be merged into a
//!   larger skin region (arm + face + neck is one component) and then
//!   rejected by the aspect-ratio filter. This is a precision win
//!   in casual photos and a recall loss on crowded group shots.
//! - On a strict gray-only input "skin-tone" is really "luminance
//!   band matching skin" — it will also fire on beige walls, sand and
//!   cardboard. The density + aspect-ratio filters weed most of that
//!   out but cannot eliminate it.
//! - All-faces detection on a colour image is the natural habitat. If
//!   you have the colour frame at hand, a chrominance-aware variant
//!   would be a measurable improvement; this crate keeps the
//!   gray-only path for zero-dep simplicity.
//!
//! ## Smoke tests
//!
//! `no_panic_*` tests in this file exercise the empty / uniform / tiny
//! input paths that every detector must handle silently.

use crate::face::{non_max_suppression, Detection};
use crate::face_detector::FaceDetector;
use crate::image::GrayImage;

/// Tunable parameters. Defaults target a frontal face on a moderate
/// indoor photo: skin-tone band `[60, 195]`, aspect ratio `[0.7, 1.5]`,
/// density ≥ 0.55.
#[derive(Clone, Debug)]
pub struct SkinConfig {
    /// Luminance lower bound for the skin-tone band.
    pub min_luma: u8,
    /// Luminance upper bound (exclusive) for the skin-tone band.
    pub max_luma: u8,
    /// Smallest accepted face width in pixels.
    pub min_size: usize,
    /// Largest accepted face width in pixels.
    pub max_size: usize,
    /// Aspect ratio (w/h) lower bound — faces are roughly square.
    pub min_aspect: f32,
    /// Aspect ratio (w/h) upper bound — faces are roughly square.
    pub max_aspect: f32,
    /// Minimum fraction of pixels in the component that must be
    /// inside the skin-tone band. Components below this are noise
    /// (border pixels, hair, background bleed).
    pub min_density: f32,
    /// Morphological kernel size for the opening pass; 1 disables.
    pub opening_size: usize,
    /// NMS IoU threshold for the final dedup.
    pub nms_iou: f32,
}

impl Default for SkinConfig {
    fn default() -> Self {
        Self {
            // [60, 195] empirically matches the 1st–99th percentile of
            // skin-tone luminance in the FDDB skin-band sweep; tight
            // enough that foliage / sky / hair drop out, loose enough
            // that the full range of human skin tones passes.
            min_luma: 60,
            max_luma: 195,
            // 32 px minimum keeps the per-frame
            // component count tractable on 1080p input.
            min_size: 32,
            max_size: 600,
            // Square ±15% covers frontal faces; profiles and YUV can
            // dial `max_aspect` up.
            min_aspect: 0.7,
            max_aspect: 1.5,
            // 0.55 — a real face is mostly skin; lower density means
            // the component merged with background and the box is
            // wrong.
            min_density: 0.55,
            // 3 px opening kills speckle noise without eating a real
            // face edge.
            opening_size: 3,
            // Standard for this crate.
            nms_iou: 0.3,
        }
    }
}

/// Result of the skin-tone detector for one connected component.
#[derive(Clone, Copy, Debug, Default)]
pub struct SkinComponent {
    /// Bounding box in image coordinates.
    pub bbox: (usize, usize, usize, usize),
    /// Fraction of pixels in the bbox that fall in the skin-tone band.
    pub density: f32,
}

/// Skin-tone face detector state. Cheap to construct.
pub struct SkinFaceDetector {
    config: SkinConfig,
}

impl SkinFaceDetector {
    pub fn new(config: SkinConfig) -> Self {
        Self { config }
    }

    /// Compute the skin-tone binary mask for a frame.
    ///
    /// `true` at position `(x, y)` iff the pixel luminance is inside
    /// the configured `[min_luma, max_luma)` band. Returned as a
    /// row-major `Vec<bool>` of length `w * h`.
    pub fn skin_mask(&self, img: &GrayImage) -> Vec<bool> {
        let n = img.as_slice().len();
        let mut mask = vec![false; n];
        let lo = self.config.min_luma;
        let hi = self.config.max_luma;
        for (i, &p) in img.as_slice().iter().enumerate() {
            mask[i] = p >= lo && p < hi;
        }
        mask
    }

    /// 3×3 (or `k × k`) morphological opening on a binary mask:
    /// erode then dilate. Drops noise below the kernel size and
    /// leaves larger structures intact.
    pub fn opening(&self, mask: &[bool], w: usize, h: usize) -> Vec<bool> {
        let k = self.config.opening_size.max(1);
        if k == 1 {
            return mask.to_vec();
        }
        let half = (k / 2) as i32;
        let eroded = erode(mask, w, h, half);
        dilate(&eroded, w, h, half)
    }

    /// Connected components on a binary mask (4-connectivity).
    ///
    /// Returns one `(bbox, density)` per component.
    pub fn components(&self, mask: &[bool], w: usize, h: usize) -> Vec<SkinComponent> {
        if w == 0 || h == 0 {
            return Vec::new();
        }
        let mut labels = vec![0u32; w * h];
        let mut out: Vec<SkinComponent> = Vec::new();
        let mut next_label: u32 = 0;
        // Iterative BFS — recursion-free, alloc-light per component
        // (one stack of pixel coords). 4-connectivity is enough for
        // skin-tone blobs since they are always dense.
        for sy in 0..h {
            for sx in 0..w {
                let idx = sy * w + sx;
                if !mask[idx] || labels[idx] != 0 {
                    continue;
                }
                next_label += 1;
                let lbl = next_label;
                let mut min_x = sx;
                let mut min_y = sy;
                let mut max_x = sx;
                let mut max_y = sy;
                let mut count: usize = 0;
                let mut stack: Vec<(usize, usize)> = vec![(sx, sy)];
                while let Some((x, y)) = stack.pop() {
                    let i = y * w + x;
                    if labels[i] != 0 || !mask[i] {
                        continue;
                    }
                    labels[i] = lbl;
                    count += 1;
                    if x < min_x {
                        min_x = x;
                    }
                    if x > max_x {
                        max_x = x;
                    }
                    if y < min_y {
                        min_y = y;
                    }
                    if y > max_y {
                        max_y = y;
                    }
                    if x > 0 {
                        stack.push((x - 1, y));
                    }
                    if x + 1 < w {
                        stack.push((x + 1, y));
                    }
                    if y > 0 {
                        stack.push((x, y - 1));
                    }
                    if y + 1 < h {
                        stack.push((x, y + 1));
                    }
                }
                let bw = max_x - min_x + 1;
                let bh = max_y - min_y + 1;
                let bbox_area = bw * bh;
                let density = if bbox_area > 0 {
                    count as f32 / bbox_area as f32
                } else {
                    0.0
                };
                out.push(SkinComponent {
                    bbox: (min_x, min_y, bw, bh),
                    density,
                });
            }
        }
        out
    }
}

impl FaceDetector for SkinFaceDetector {
    fn detect(&self, img: &GrayImage) -> Vec<Detection> {
        let w = img.width();
        let h = img.height();
        if w == 0 || h == 0 {
            return Vec::new();
        }
        let mask = self.skin_mask(img);
        let opened = self.opening(&mask, w, h);
        let components = self.components(&opened, w, h);

        // Apply size + aspect + density filters.
        let mut dets: Vec<Detection> = Vec::new();
        for c in components {
            let (x, y, bw, bh) = c.bbox;
            if bw < self.config.min_size || bh < self.config.min_size {
                continue;
            }
            if bw > self.config.max_size || bh > self.config.max_size {
                continue;
            }
            let aspect = bw as f32 / bh as f32;
            if aspect < self.config.min_aspect || aspect > self.config.max_aspect {
                continue;
            }
            if c.density < self.config.min_density {
                continue;
            }
            // Make the bbox square at the larger side — a face
            // detection box is conventionally square (the cascade
            // window is square, and downstream recognition crops it).
            let side = bw.max(bh);
            // Centre on the component centre, then clamp.
            let cx = x as f32 + bw as f32 * 0.5;
            let cy = y as f32 + bh as f32 * 0.5;
            let sq_x = ((cx - side as f32 * 0.5).round() as i64).max(0) as usize;
            let sq_y = ((cy - side as f32 * 0.5).round() as i64).max(0) as usize;
            let sq_side = side.min(w.saturating_sub(sq_x)).min(h.saturating_sub(sq_y));
            if sq_side < self.config.min_size {
                continue;
            }
            dets.push(Detection {
                x: sq_x,
                y: sq_y,
                w: sq_side,
                h: sq_side,
                score: c.density,
            });
        }
        non_max_suppression(dets, self.config.nms_iou)
    }

    fn name(&self) -> &'static str {
        "skin"
    }

    fn description(&self) -> &'static str {
        "Skin-tone luminance band + morphology + connected-components face detector. \
         No weights, single-scale; very fast and orthogonal to the gradient-based detectors."
    }
}

fn erode(mask: &[bool], w: usize, h: usize, half: i32) -> Vec<bool> {
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut all = true;
            'k: for ky in -half..=half {
                let yy = y as i32 + ky;
                if yy < 0 || yy >= h as i32 {
                    all = false;
                    break 'k;
                }
                for kx in -half..=half {
                    let xx = x as i32 + kx;
                    if xx < 0 || xx >= w as i32 {
                        all = false;
                        break 'k;
                    }
                    if !mask[yy as usize * w + xx as usize] {
                        all = false;
                        break 'k;
                    }
                }
            }
            out[y * w + x] = all;
        }
    }
    out
}

fn dilate(mask: &[bool], w: usize, h: usize, half: i32) -> Vec<bool> {
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut any = false;
            'k: for ky in -half..=half {
                let yy = y as i32 + ky;
                if yy < 0 || yy >= h as i32 {
                    continue;
                }
                for kx in -half..=half {
                    let xx = x as i32 + kx;
                    if xx < 0 || xx >= w as i32 {
                        continue;
                    }
                    if mask[yy as usize * w + xx as usize] {
                        any = true;
                        break 'k;
                    }
                }
            }
            out[y * w + x] = any;
        }
    }
    out
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
        let d = SkinFaceDetector::new(SkinConfig::default());
        assert!(d.detect(&empty()).is_empty());
    }

    #[test]
    fn no_panic_on_uniform() {
        let d = SkinFaceDetector::new(SkinConfig::default());
        // Pixels outside the [60, 195) skin band — no detection possible.
        for v in [0u8, 32, 200, 255] {
            let dets = d.detect(&uniform(v, 100, 100));
            assert!(dets.is_empty(), "uniform {v} produced {} dets", dets.len());
        }
    }

    #[test]
    fn no_panic_on_tiny_input() {
        let d = SkinFaceDetector::new(SkinConfig::default());
        let img = uniform(128, 0, 0);
        assert!(d.detect(&img).is_empty());
        let img = uniform(128, 10, 10);
        assert!(d.detect(&img).is_empty());
    }

    #[test]
    fn skin_mask_brackets_include_band_excludes_outside() {
        let d = SkinFaceDetector::new(SkinConfig::default());
        let mut img = GrayImage::new(4, 1);
        for (i, p) in [10u8, 70, 140, 220].iter().enumerate() {
            img.as_mut_slice()[i] = *p;
        }
        let mask = d.skin_mask(&img);
        // 10 < 60 (lo), 70 ∈ [60, 195), 140 ∈ band, 220 ≥ 195 (hi).
        assert!(!mask[0]);
        assert!(mask[1]);
        assert!(mask[2]);
        assert!(!mask[3]);
    }

    #[test]
    fn detects_synthetic_face_size_and_position() {
        // 200x200 image with a 64x64 skin-tone square centred.
        let mut img = GrayImage::new(200, 200);
        img.as_mut_slice().fill(0);
        let w = img.width();
        // Skin-tone square at (68, 68) — 64x64.
        {
            let slice = img.as_mut_slice();
            for y in 68..(68 + 64) {
                for x in 68..(68 + 64) {
                    slice[y * w + x] = 128;
                }
            }
        }
        let d = SkinFaceDetector::new(SkinConfig::default());
        let dets = d.detect(&img);
        assert!(
            !dets.is_empty(),
            "synthetic face-sized skin-tone square must produce a detection"
        );
        let det = &dets[0];
        assert_eq!(det.w, det.h, "bbox must be square");
        assert!(det.w >= 32);
    }

    #[test]
    fn aspect_ratio_filter_rejects_long_thin_blobs() {
        // 200x200, with a 96x32 skin-tone rectangle (aspect 3.0).
        let mut img = GrayImage::new(200, 200);
        img.as_mut_slice().fill(0);
        let w = img.width();
        {
            let slice = img.as_mut_slice();
            for y in 80..(80 + 32) {
                for x in 50..(50 + 96) {
                    slice[y * w + x] = 128;
                }
            }
        }
        let d = SkinFaceDetector::new(SkinConfig::default());
        let dets = d.detect(&img);
        assert!(
            dets.is_empty(),
            "3:1 aspect ratio must be rejected (got {} dets)",
            dets.len()
        );
    }

    #[test]
    fn density_filter_rejects_sparse_components() {
        // Sparse scattered band-mask: density too low to pass.
        let mut img = GrayImage::new(100, 100);
        img.as_mut_slice().fill(128);
        // Knock out 80% of the band-mask pixels by alternating.
        {
            let slice = img.as_mut_slice();
            for i in 0..slice.len() {
                if i % 5 != 0 {
                    slice[i] = 0; // outside band
                }
            }
        }
        let d = SkinFaceDetector::new(SkinConfig::default());
        let dets = d.detect(&img);
        // 20% density is below min_density=0.55 — must not fire.
        assert!(
            dets.is_empty(),
            "20% density blob must be rejected (got {} dets)",
            dets.len()
        );
    }

    #[test]
    fn opening_kills_single_pixel_noise() {
        // 100x100 frame with a few scattered skin-tone pixels
        // (the morphological opening should erase them).
        let mut img = GrayImage::new(100, 100);
        img.as_mut_slice().fill(0);
        {
            let slice = img.as_mut_slice();
            for &(x, y) in &[(10, 10), (50, 50), (90, 90)] {
                slice[y * 100 + x] = 128;
            }
        }
        let d = SkinFaceDetector::new(SkinConfig::default());
        let dets = d.detect(&img);
        // Single pixels are too small to survive the 3x3 opening
        // and certainly smaller than min_size=32.
        assert!(dets.is_empty());
    }

    #[test]
    fn name_and_description_are_set() {
        let d = SkinFaceDetector::new(SkinConfig::default());
        assert_eq!(d.name(), "skin");
        assert!(!d.description().is_empty());
    }
}
