//! Multi-scale sliding window detector + non-maximum suppression.

use crate::haar::{Cascade, EvalCache};
use crate::image::GrayImage;
use crate::integral::{IntegralImage, RotatedIntegralImage, SquaredIntegralImage};

/// A single detection: pixel-space bounding box + confidence score.
#[derive(Clone, Debug)]
pub struct Detection {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub score: f32,
}

/// Configuration for the detector.
#[derive(Clone, Debug)]
pub struct DetectorConfig {
    /// Minimum detection size in pixels (width). The image is downscaled until
    /// this width is reached.
    pub min_size: usize,
    /// Maximum detection size in pixels. Detections larger than the image are
    /// clamped.
    pub max_size: usize,
    /// Scale factor between successive pyramid levels (< 1.0 = zoom in).
    pub scale_factor: f32,
    /// Window stride in pixels at the original image scale.
    pub window_stride: usize,
    /// Final NMS IoU threshold; overlapping detections above this are merged.
    pub nms_iou_threshold: f32,
    /// Cascade score threshold — detections below this are dropped.
    pub min_score: f32,
    /// Variance pre-filter threshold. Windows whose variance is below this
    /// value are skipped without evaluating the cascade. Set to `u64::MAX` to
    /// disable. The default of 200 corresponds roughly to OpenCV's default
    /// (which uses `minEig = 4000` for the 24x24 window — we use 1/20th of that
    /// since our variance calculation is on the same scale as `var = E[X²] - E[X]²`).
    pub variance_threshold: u64,
    /// If `true`, attempt to use the GPU for the squared-integral computation
    /// and variance pre-filter. Falls back to CPU silently if no GPU/OpenCL
    /// is available.
    pub use_gpu: bool,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            min_size: 24,
            max_size: 1024,
            scale_factor: 1.2,
            window_stride: 4,
            nms_iou_threshold: 0.3,
            min_score: 0.0,
            variance_threshold: 200,
            use_gpu: true,
        }
    }
}

pub struct Detector {
    pub cascade: Cascade,
    pub config: DetectorConfig,
    /// Optional GPU context. Lazily initialized.
    gpu: std::sync::OnceLock<Option<crate::gpu::GpuIntegral>>,
}

impl Detector {
    pub fn new(cascade: Cascade, config: DetectorConfig) -> Self {
        Self { cascade, config, gpu: std::sync::OnceLock::new() }
    }

    fn gpu(&self) -> Option<&crate::gpu::GpuIntegral> {
        if !self.config.use_gpu { return None; }
        self.gpu.get_or_init(|| crate::gpu::GpuIntegral::new()).as_ref()
    }

    /// True when GPU is worth invoking for an image of this size. GPU kernel
    /// launch + PCIe transfer has fixed overhead; below ~500×500 the CPU
    /// wins. Threshold tuned on this Tegra-class hardware.
    fn gpu_worthwhile(img_w: usize, img_h: usize) -> bool {
        img_w * img_h >= 500 * 500
    }

    /// Detect faces in a grayscale image. Returns a vector of detections
    /// sorted by descending score.
    pub fn detect(&self, img: &GrayImage) -> Vec<Detection> {
        let mut raw: Vec<Detection> = Vec::new();
        let win_w = self.cascade.window_w;
        let win_h = self.cascade.window_h;
        if img.width() < win_w || img.height() < win_h { return raw; }

        // Per-thread scratch buffer. One allocation per Detector, reused for
        // every window — eliminated the previous `vec![None; 2913]` per-window
        // allocation that was the dominant cost.
        let mut cache = EvalCache::new(self.cascade.features.len());

        // Build pyramid by repeated downscaling.
        let mut current = img.clone();
        let mut current_scale: f32 = 1.0;
        loop {
            let cw = current.width();
            let ch = current.height();
            let det_w_at_cur = (win_w as f32 * current_scale).round() as usize;
            let det_h_at_cur = (win_h as f32 * current_scale).round() as usize;
            if det_w_at_cur > self.config.max_size || det_h_at_cur > self.config.max_size { break; }
            if det_w_at_cur < self.config.min_size || det_h_at_cur < self.config.min_size { break; }
            if cw < win_w || ch < win_h { break; }

            // Build integral images. The squared integral is rebuilt per-level
            // because variance normalisation must use the SAME pixels as the
            // feature responses (i.e. the current pyramid level). On GPU we
            // get both for free in one pass; on CPU we make them separately.
            let (ii, ii_sq) = if let Some(g) = self.gpu() {
                if Self::gpu_worthwhile(cw, ch) {
                    let (ii_data, ii_sq_data) = g.compute_dual(&current);
                    (IntegralImage::from_owned(ii_data, cw, ch),
                     SquaredIntegralImage::from_owned(ii_sq_data, cw, ch))
                } else {
                    (IntegralImage::from_gray(&current),
                     SquaredIntegralImage::from_gray(&current))
                }
            } else {
                (IntegralImage::from_gray(&current),
                 SquaredIntegralImage::from_gray(&current))
            };
            // Move the squared integral image into the cache without an
            // intermediate clone (previously this was `cached_sq.clone()`
            // which copied the entire (W+1)*(H+1) u64 buffer).
            cache.set_squared_iis(ii_sq);
            let ri = RotatedIntegralImage::from_gray(&current);
            // Integer-scaled stride: stride_at_scale = base_stride * current_scale
            // approximated by the nearest integer; the (.max(2)) keeps us from
            // sliding window on every pixel for big detections.
            let stride = ((self.config.window_stride as f32) * current_scale)
                .round()
                .max(2.0) as usize;
            let use_variance = self.config.variance_threshold < u64::MAX;

            // GPU fast-path: run the full cascade on GPU when worth it.
            // The kernel handles variance normalisation + per-stage eval +
            // early rejection in parallel across all (x, y) windows.
            if stride == 1 {
                if let Some(g) = self.gpu() {
                    if Self::gpu_worthwhile(cw, ch) {
                        let max_dets = ((cw - win_w + 1) * (ch - win_h + 1)).min(8192);
                        let gpu_dets = g.detect_windows(&self.cascade, &current, max_dets);
                        for d in gpu_dets {
                            if d.score < self.config.min_score { continue; }
                            let ox = (d.x as f32 / current_scale).round() as usize;
                            let oy = (d.y as f32 / current_scale).round() as usize;
                            let ox = ox.min(img.width().saturating_sub(det_w_at_cur));
                            let oy = oy.min(img.height().saturating_sub(det_h_at_cur));
                            raw.push(Detection { x: ox, y: oy, w: det_w_at_cur, h: det_h_at_cur, score: d.score });
                        }
                    }
                }
            }

            let mut y = 0;
            while y + win_h <= ch {
                let mut x = 0;
                while x + win_w <= cw {
                    // Variance pre-filter: cheap O(1) rejection of windows that
                    // cannot contain a face. Caches rejects the vast majority of
                    // windows in real images and saves the full cascade evaluation.
                    let passes = !use_variance || cache.passes_variance(
                        &ii, x, y, win_w, win_h, self.config.variance_threshold);
                    if passes {
                        if let Some(score) = self.cascade.classify(&ii, &ri, x, y, &mut cache) {
                            if score >= self.config.min_score {
                                // Map (x, y) at current scale back to original image space.
                                let ox = (x as f32 / current_scale).round() as usize;
                                let oy = (y as f32 / current_scale).round() as usize;
                                let ow = det_w_at_cur;
                                let oh = det_h_at_cur;
                                // Clamp to image bounds.
                                let ox = ox.min(img.width().saturating_sub(ow));
                                let oy = oy.min(img.height().saturating_sub(oh));
                                raw.push(Detection { x: ox, y: oy, w: ow, h: oh, score });
                            }
                        }
                    }
                    x += stride;
                }
                y += stride;
            }

            // Prepare next pyramid level.
            let next_w = ((cw as f32) / self.config.scale_factor).round().max(win_w as f32) as usize;
            let next_h = ((ch as f32) / self.config.scale_factor).round().max(win_h as f32) as usize;
            if next_w == cw || next_h == ch { break; }
            // For downscaling (next_w < cw), use area averaging which matches
            // OpenCV's default `cv::resize` for >2× downscaling and is significantly
            // more accurate than bilinear for cascade evaluation.
            if next_w < cw {
                current = current.resize_area(next_w, next_h);
            } else {
                current = current.resize_bilinear(next_w, next_h);
            }
            current_scale = img.width() as f32 / current.width() as f32;
            if current.width() <= win_w || current.height() <= win_h { break; }
        }

        non_max_suppression(raw, self.config.nms_iou_threshold)
    }
}

/// Standard greedy NMS: pick the highest-score box, suppress all with IoU > threshold.
pub fn non_max_suppression(mut dets: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<Detection> = Vec::new();
    let mut suppressed = vec![false; dets.len()];
    for i in 0..dets.len() {
        if suppressed[i] { continue; }
        keep.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if suppressed[j] { continue; }
            if iou(&dets[i], &dets[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

#[inline]
fn iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let w = (x2 as i64 - x1 as i64).max(0) as usize;
    let h = (y2 as i64 - y1 as i64).max(0) as usize;
    let inter = (w * h) as f32;
    if inter <= 0.0 { return 0.0; }
    let union = (a.w * a.h + b.w * b.h) as f32 - inter;
    if union <= 0.0 { return 0.0; }
    inter / union
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::haar::params::demo_face_cascade;

    #[test]
    fn detects_bright_center_in_uniform_image() {
        let mut img = GrayImage::new(120, 120);
        for y in 0..120 {
            for x in 0..120 {
                let v = if y < 20 { 20 }
                       else if y < 40 && (40..80).contains(&x) { 200 }
                       else if y < 100 && (40..80).contains(&x) { 220 }
                       else { 20 };
                img[(x, y)] = v;
            }
        }
        let det = Detector::new(demo_face_cascade(), DetectorConfig::default());
        let r = det.detect(&img);
        assert!(!r.is_empty(), "expected at least one detection in bright-center pattern");
    }

    #[test]
    fn nsm_merges_overlapping() {
        let a = Detection { x: 0, y: 0, w: 20, h: 20, score: 1.0 };
        let b = Detection { x: 2, y: 2, w: 20, h: 20, score: 0.9 };
        let c = Detection { x: 100, y: 100, w: 20, h: 20, score: 0.5 };
        let r = non_max_suppression(vec![a.clone(), b.clone(), c.clone()], 0.3);
        assert_eq!(r.len(), 2, "should merge a+b but keep c");
        assert_eq!(r[0].score, 1.0);
    }
}
