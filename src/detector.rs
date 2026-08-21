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

impl Detection {
    /// Right edge (exclusive) of the bounding box.
    #[inline]
    pub fn right(&self) -> usize {
        self.x + self.w
    }

    /// Bottom edge (exclusive) of the bounding box.
    #[inline]
    pub fn bottom(&self) -> usize {
        self.y + self.h
    }

    /// Area in pixels (`w * h`).
    #[inline]
    pub fn area(&self) -> usize {
        self.w * self.h
    }

    /// Intersection-over-union with another detection, in `[0, 1]`.
    /// Overlapping boxes return `> 0`; disjoint boxes return exactly `0`.
    pub fn iou(&self, other: &Detection) -> f32 {
        iou(self, other)
    }

    /// Center point `(cx, cy)` of the bounding box.
    #[inline]
    pub fn center(&self) -> (f32, f32) {
        (
            self.x as f32 + self.w as f32 / 2.0,
            self.y as f32 + self.h as f32 / 2.0,
        )
    }
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
    /// If `true`, apply `cv::equalizeHist`-style histogram equalization to the
    /// image before running the cascade. OpenCV's Haar cascade is trained on
    /// equalized data — without this, real photographs with low contrast or
    /// shifted luminance get silently rejected at stage 0 because the
    /// per-feature thresholds were calibrated for the equalized range.
    /// Defaults to `true`; set to `false` to compare with raw input.
    pub equalize_hist: bool,
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
            equalize_hist: false,
            use_gpu: true,
        }
    }
}

impl DetectorConfig {
    /// Preset tuned for throughput (e.g. live video preview): coarser pyramid
    /// (1.25×), stride 6 and a lighter variance pre-filter. Trades a little
    /// recall on small faces for roughly 2× fewer window evaluations than
    /// [`DetectorConfig::default`].
    pub fn fast() -> Self {
        Self {
            scale_factor: 1.25,
            window_stride: 6,
            variance_threshold: 300,
            ..Self::default()
        }
    }

    /// Preset tuned for accuracy (offline batch): finer pyramid (1.1×),
    /// stride 2, full-range variance pre-filter. Roughly 4× more window
    /// evaluations than [`DetectorConfig::default`]; finds smaller faces and
    /// tighter boxes.
    pub fn accurate() -> Self {
        Self {
            scale_factor: 1.1,
            window_stride: 2,
            variance_threshold: 100,
            ..Self::default()
        }
    }

    /// Clamp `min_size`/`max_size` into a sane order for `image`
    /// (`min_size <= max_size`, `min_size >= 1`) and the pyramid geometry
    /// (`scale_factor` in `[1.01, 2.0]`, stride `>= 1`). Useful after the
    /// struct has been assembled from untrusted CLI/JSON input — the
    /// constructor fields stay public, this just repairs nonsense values.
    pub fn sanitized(mut self) -> Self {
        self.min_size = self.min_size.max(1);
        self.max_size = self.max_size.max(self.min_size);
        if !self.scale_factor.is_finite() || self.scale_factor < 1.01 {
            self.scale_factor = 1.01;
        }
        if self.scale_factor > 2.0 {
            self.scale_factor = 2.0;
        }
        self.window_stride = self.window_stride.max(1);
        self
    }

    /// Builder-style override of `min_size` (see [`Self::sanitized`] for
    /// the full validation pass).
    pub fn with_min_size(mut self, min_size: usize) -> Self {
        self.min_size = min_size;
        self
    }

    /// Builder-style override of `max_size`.
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }

    /// Builder-style override of `scale_factor`.
    pub fn with_scale_factor(mut self, scale_factor: f32) -> Self {
        self.scale_factor = scale_factor;
        self
    }

    /// Builder-style override of `window_stride`.
    pub fn with_window_stride(mut self, window_stride: usize) -> Self {
        self.window_stride = window_stride;
        self
    }

    /// Builder-style override of `nms_iou_threshold`.
    pub fn with_nms_iou_threshold(mut self, nms_iou_threshold: f32) -> Self {
        self.nms_iou_threshold = nms_iou_threshold;
        self
    }

    /// Builder-style override of `min_score`.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// Builder-style toggle of the GPU path.
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Builder-style toggle of histogram equalisation.
    pub fn with_equalize_hist(mut self, equalize_hist: bool) -> Self {
        self.equalize_hist = equalize_hist;
        self
    }

    /// Builder-style override of `variance_threshold` (`u64::MAX` disables
    /// the pre-filter).
    pub fn with_variance_threshold(mut self, variance_threshold: u64) -> Self {
        self.variance_threshold = variance_threshold;
        self
    }
}

/// Outcome of a single [`Detector::detect_timed`] call: the surviving
/// detections plus per-frame instrumentation.
#[derive(Clone, Debug)]
pub struct DetectResult {
    /// Detections after NMS, sorted by descending score (same order as
    /// [`Detector::detect`]).
    pub detections: Vec<Detection>,
    /// Wall-clock detection time in milliseconds (pyramid + integrals +
    /// cascade + NMS).
    pub detect_ms: f64,
    /// Number of pyramid levels scanned.
    pub levels: usize,
    /// Number of sliding-window positions visited (post-stride).
    pub windows_evaluated: usize,
}

pub struct Detector {
    pub cascade: Cascade,
    pub config: DetectorConfig,
    /// Optional GPU context. Lazily initialized.
    gpu: std::sync::OnceLock<Option<crate::gpu::GpuIntegral>>,
    /// Computed once in `new()`: does any cascade feature need the rotated
    /// (45°) integral image? If not, its per-level construction is skipped.
    needs_tilted: bool,
}

impl Detector {
    pub fn new(cascade: Cascade, config: DetectorConfig) -> Self {
        let needs_tilted = cascade
            .features
            .iter()
            .any(|f| matches!(f.kind, crate::haar::FeatureKind::DiagonalEdge));
        Self {
            cascade,
            config,
            gpu: std::sync::OnceLock::new(),
            needs_tilted,
        }
    }

    /// Whether any feature of the loaded cascade uses tilted (45°) rects.
    /// When false, the rotated integral image is never built — the full
    /// `(W+1)×(H+1)` i64 table (the most expensive integral of the three)
    /// is skipped at every pyramid level.
    pub fn needs_tilted(&self) -> bool {
        self.needs_tilted
    }

    fn gpu(&self) -> Option<&crate::gpu::GpuIntegral> {
        if !self.config.use_gpu {
            return None;
        }
        self.gpu
            .get_or_init(|| crate::gpu::GpuIntegral::new())
            .as_ref()
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
        self.detect_inner(img).0
    }

    /// [`Self::detect`] with per-frame instrumentation (see [`DetectResult`]).
    /// The detection list is byte-identical to `detect(img)`; only the
    /// bookkeeping is added.
    pub fn detect_timed(&self, img: &GrayImage) -> DetectResult {
        let start = std::time::Instant::now();
        let (dets, levels, windows) = self.detect_inner(img);
        DetectResult {
            detections: dets,
            detect_ms: start.elapsed().as_secs_f64() * 1e3,
            levels,
            windows_evaluated: windows,
        }
    }

    /// Core scan. Returns `(detections, levels, windows_evaluated)`.
    fn detect_inner(&self, img: &GrayImage) -> (Vec<Detection>, usize, usize) {
        let mut raw: Vec<Detection> = Vec::new();
        let mut levels = 0usize;
        let mut windows_evaluated = 0usize;
        let win_w = self.cascade.window_w;
        let win_h = self.cascade.window_h;
        if img.width() < win_w || img.height() < win_h {
            return (raw, levels, windows_evaluated);
        }

        // Per-thread scratch buffer. One allocation per Detector, reused for
        // every window — eliminated the previous `vec![None; 2913]` per-window
        // allocation that was the dominant cost.
        let mut cache = EvalCache::new(self.cascade.features.len());

        // Build pyramid by repeated downscaling.
        // OpenCV's Haar cascade is trained on histogram-equalized images
        // (the canonical "Lena face detector" workflow applies
        // `cv::equalizeHist` before `detectMultiScale`). Without it the
        // integral-image sums land in a different numerical range than the
        // cascade's learned thresholds/varianceNormFactor and most real
        // faces silently fail stage 0. The equalization is a deterministic
        // O(W*H) per pixel pass — cheap relative to the cascade eval.
        let eq_storage: Option<GrayImage> = if self.config.equalize_hist {
            let mut eq = img.clone();
            eq.equalize_hist_inplace();
            Some(eq)
        } else {
            None
        };
        let current: &GrayImage = eq_storage.as_ref().unwrap_or(img);
        let mut downscaled: Option<GrayImage> = None;
        let mut current_scale: f32 = 1.0;
        loop {
            // `current` is the original (or equalized) image on the first
            // iteration and the owned pyramid level afterwards.
            let current: &GrayImage = downscaled.as_ref().unwrap_or(current);
            let cw = current.width();
            let ch = current.height();
            let det_w_at_cur = (win_w as f32 * current_scale).round() as usize;
            let det_h_at_cur = (win_h as f32 * current_scale).round() as usize;
            if det_w_at_cur > self.config.max_size || det_h_at_cur > self.config.max_size {
                break;
            }
            if det_w_at_cur < self.config.min_size || det_h_at_cur < self.config.min_size {
                break;
            }
            if cw < win_w || ch < win_h {
                break;
            }
            levels += 1;

            // Build integral images. The squared integral is rebuilt per-level
            // because variance normalisation must use the SAME pixels as the
            // feature responses (i.e. the current pyramid level). On GPU we
            // get both for free in one pass; on CPU we make them separately.
            let (ii, ii_sq) = if let Some(g) = self.gpu() {
                if Self::gpu_worthwhile(cw, ch) {
                    let (ii_data, ii_sq_data) = g.compute_dual(current);
                    (
                        IntegralImage::from_owned(ii_data, cw, ch),
                        SquaredIntegralImage::from_owned(ii_sq_data, cw, ch),
                    )
                } else {
                    (
                        IntegralImage::from_gray(current),
                        SquaredIntegralImage::from_gray(current),
                    )
                }
            } else {
                (
                    IntegralImage::from_gray(current),
                    SquaredIntegralImage::from_gray(current),
                )
            };
            // Move the squared integral image into the cache without an
            // intermediate clone (previously this was `cached_sq.clone()`
            // which copied the entire (W+1)*(H+1) u64 buffer).
            cache.set_squared_iis(ii_sq);
            // Rotated integral: only cascades with tilted (DiagonalEdge)
            // features ever query it. The demo cascade has none, so skip the
            // (W+1)×(H+1) i64 construction entirely for such cascades.
            let ri = if self.needs_tilted {
                RotatedIntegralImage::from_gray(current)
            } else {
                RotatedIntegralImage::empty()
            };
            // Integer-scaled stride: stride_at_scale = base_stride * current_scale
            // approximated by the nearest integer; the (.max(2)) keeps us from
            // sliding window on every pixel for big detections.
            let stride = ((self.config.window_stride as f32) * current_scale)
                .round()
                .max(2.0) as usize;
            let use_variance = self.config.variance_threshold < u64::MAX;

            // GPU fast-path: run the full cascade on GPU when worth it.
            // The kernel handles variance normalisation + per-stage eval +        // early rejection in parallel across all (x, y) windows.
            if stride == 1 {
                if let Some(g) = self.gpu() {
                    if Self::gpu_worthwhile(cw, ch) {
                        let max_dets = ((cw - win_w + 1) * (ch - win_h + 1)).min(8192);
                        let gpu_dets = g.detect_windows(&self.cascade, current, max_dets);
                        for d in gpu_dets {
                            if d.score < self.config.min_score {
                                continue;
                            }
                            let ox = (d.x as f32 / current_scale).round() as usize;
                            let oy = (d.y as f32 / current_scale).round() as usize;
                            let ox = ox.min(img.width().saturating_sub(det_w_at_cur));
                            let oy = oy.min(img.height().saturating_sub(det_h_at_cur));
                            raw.push(Detection {
                                x: ox,
                                y: oy,
                                w: det_w_at_cur,
                                h: det_h_at_cur,
                                score: d.score,
                            });
                        }
                    }
                }
            }

            let mut y = 0;
            while y + win_h <= ch {
                let mut x = 0;
                while x + win_w <= cw {
                    windows_evaluated += 1;
                    // Variance pre-filter: cheap O(1) rejection of windows that
                    // cannot contain a face. Rejects the vast majority of
                    // windows in real images and saves the full cascade
                    // evaluation.
                    //
                    // Mirrors OpenCV's `HaarEvaluator::setWindow` early-exit: the
                    // variance is computed over the INNER normrect (1, 1, W-2, H-2)
                    // — the same area the cascade's `varianceNormFactor` is built
                    // from — so a window that passes the pre-filter is guaranteed
                    // to have a positive normrect variance when the cascade
                    // evaluates it (no wasted `variance_norm_factor == 0` rejects).
                    //
                    // The window fits the image (`x + win_w <= cw`) so the
                    // inner rect fits too; use clamp-free reads and hand
                    // the pre-filter's (sum, sum_sq) straight into the fused
                    // classify, which needs the same pair for
                    // `varianceNormFactor` — one pair of rectangle reads
                    // instead of two.
                    //
                    // SAFETY (rect_sum_unchecked): the inner normrect
                    // [x+1, x+win_w-1) × [y+1, y+win_h-1) is strictly inside
                    // the window, and the loop guards guarantee the window
                    // fits the level image (x + win_w <= cw, y + win_h <= ch).
                    let score_opt = if use_variance {
                        let (s, ss) = unsafe {
                            (
                                ii.rect_sum_unchecked(x + 1, y + 1, x + win_w - 1, y + win_h - 1),
                                cache.sum_sq_rect_sum_unchecked(
                                    x + 1,
                                    y + 1,
                                    x + win_w - 1,
                                    y + win_h - 1,
                                ),
                            )
                        };
                        if !SquaredIntegralImage::passes_variance_sums(
                            s,
                            ss,
                            win_w - 2,
                            win_h - 2,
                            self.config.variance_threshold,
                        ) {
                            None
                        } else {
                            self.cascade.classify_inbounds_with_sums(
                                &ii,
                                &ri,
                                x,
                                y,
                                &mut cache,
                                (s, ss),
                            )
                        }
                    } else {
                        self.cascade.classify_inbounds(&ii, &ri, x, y, &mut cache)
                    };
                    if let Some(score) = score_opt {
                        if score >= self.config.min_score {
                            // Map (x, y) at current scale back to original image space.
                            let ox = (x as f32 / current_scale).round() as usize;
                            let oy = (y as f32 / current_scale).round() as usize;
                            let ow = det_w_at_cur;
                            let oh = det_h_at_cur;
                            // Clamp to image bounds.
                            let ox = ox.min(img.width().saturating_sub(ow));
                            let oy = oy.min(img.height().saturating_sub(oh));
                            raw.push(Detection {
                                x: ox,
                                y: oy,
                                w: ow,
                                h: oh,
                                score,
                            });
                        }
                    }
                    x += stride;
                }
                y += stride;
            }

            // Prepare next pyramid level.
            let next_w = ((cw as f32) / self.config.scale_factor)
                .round()
                .max(win_w as f32) as usize;
            let next_h = ((ch as f32) / self.config.scale_factor)
                .round()
                .max(win_h as f32) as usize;
            if next_w == cw || next_h == ch {
                break;
            }
            // For downscaling (next_w < cw), use area averaging which matches
            // OpenCV's default `cv::resize` for >2× downscaling and is significantly
            // more accurate than bilinear for cascade evaluation.
            if next_w < cw {
                downscaled = Some(current.resize_area(next_w, next_h));
            } else {
                downscaled = Some(current.resize_bilinear(next_w, next_h));
            }
            let next = downscaled.as_ref().expect("just stored");
            current_scale = img.width() as f32 / next.width() as f32;
            if next.width() <= win_w || next.height() <= win_h {
                break;
            }
        }

        (
            non_max_suppression(raw, self.config.nms_iou_threshold),
            levels,
            windows_evaluated,
        )
    }
}

/// Standard greedy NMS: pick the highest-score box, suppress all with IoU > threshold.
pub fn non_max_suppression(mut dets: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep: Vec<Detection> = Vec::new();
    let mut suppressed = vec![false; dets.len()];
    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if suppressed[j] {
                continue;
            }
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
    if inter <= 0.0 {
        return 0.0;
    }
    let union = (a.w * a.h + b.w * b.h) as f32 - inter;
    if union <= 0.0 {
        return 0.0;
    }
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
                let v = if y < 20 {
                    20
                } else if y < 40 && (40..80).contains(&x) {
                    200
                } else if y < 100 && (40..80).contains(&x) {
                    220
                } else {
                    20
                };
                img[(x, y)] = v;
            }
        }
        let det = Detector::new(demo_face_cascade(), DetectorConfig::default());
        let r = det.detect(&img);
        assert!(
            !r.is_empty(),
            "expected at least one detection in bright-center pattern"
        );
    }

    #[test]
    fn nsm_merges_overlapping() {
        let a = Detection {
            x: 0,
            y: 0,
            w: 20,
            h: 20,
            score: 1.0,
        };
        let b = Detection {
            x: 2,
            y: 2,
            w: 20,
            h: 20,
            score: 0.9,
        };
        let c = Detection {
            x: 100,
            y: 100,
            w: 20,
            h: 20,
            score: 0.5,
        };
        let r = non_max_suppression(vec![a.clone(), b.clone(), c.clone()], 0.3);
        assert_eq!(r.len(), 2, "should merge a+b but keep c");
        assert_eq!(r[0].score, 1.0);
    }

    #[test]
    fn detection_helpers() {
        let a = Detection {
            x: 10,
            y: 20,
            w: 30,
            h: 40,
            score: 0.75,
        };
        assert_eq!(a.right(), 40);
        assert_eq!(a.bottom(), 60);
        assert_eq!(a.area(), 1200);
        let (cx, cy) = a.center();
        assert_eq!(cx, 25.0);
        assert_eq!(cy, 40.0);
        // Identical box → IoU 1; disjoint → 0.
        let b = a.clone();
        assert!((a.iou(&b) - 1.0).abs() < 1e-6);
        let c = Detection {
            x: 200,
            y: 200,
            w: 10,
            h: 10,
            score: 0.1,
        };
        assert_eq!(a.iou(&c), 0.0);
    }

    #[test]
    fn detector_knows_tilted_needs() {
        // The demo cascade has NO DiagonalEdge feature → the rotated
        // integral image is never queried, so its construction is skipped.
        let det = Detector::new(demo_face_cascade(), DetectorConfig::default());
        assert!(!det.needs_tilted());

        // A cascade containing a DiagonalEdge feature must request it.
        let mut c = demo_face_cascade();
        c.features
            .push(crate::haar::HaarFeature::diagonal_edge(2, 2));
        let idx = (c.features.len() - 1) as u32;
        c.stages.push(crate::haar::Stage {
            stage_threshold: -10.0,
            weak_features: vec![crate::haar::WeakFeature {
                feature_index: idx,
                threshold: 0.0,
                sign: 1,
                left_val: 0.0,
                right_val: 0.0,
            }],
        });
        let det = Detector::new(c, DetectorConfig::default());
        assert!(det.needs_tilted());
    }

    #[test]
    fn config_presets_and_builders() {
        let fast = DetectorConfig::fast();
        assert_eq!(fast.window_stride, 6);
        assert!(fast.scale_factor > DetectorConfig::default().scale_factor);
        let acc = DetectorConfig::accurate();
        assert_eq!(acc.window_stride, 2);
        assert!(acc.scale_factor < DetectorConfig::default().scale_factor);

        let built = DetectorConfig::default()
            .with_min_size(32)
            .with_max_size(512)
            .with_window_stride(3)
            .with_gpu(false)
            .with_equalize_hist(true)
            .with_min_score(1.5)
            .with_nms_iou_threshold(0.5)
            .with_scale_factor(1.15)
            .with_variance_threshold(12345);
        assert_eq!(built.min_size, 32);
        assert_eq!(built.max_size, 512);
        assert_eq!(built.window_stride, 3);
        assert!(!built.use_gpu);
        assert!(built.equalize_hist);
        assert_eq!(built.min_score, 1.5);
        assert_eq!(built.nms_iou_threshold, 0.5);
        assert_eq!(built.scale_factor, 1.15);
        assert_eq!(built.variance_threshold, 12345);

        // sanitized() repairs nonsense input.
        let broken = DetectorConfig {
            min_size: 400,
            max_size: 24,
            scale_factor: 0.5,
            window_stride: 0,
            ..DetectorConfig::default()
        }
        .sanitized();
        assert_eq!(broken.min_size, 400);
        assert_eq!(broken.max_size, 400);
        assert_eq!(broken.scale_factor, 1.01);
        assert_eq!(broken.window_stride, 1);
    }

    #[test]
    fn detect_timed_matches_detect() {
        let mut img = GrayImage::new(96, 96);
        for y in 0..96 {
            for x in 0..96 {
                let v = if (20..80).contains(&y) && (30..70).contains(&x) {
                    220
                } else {
                    20
                };
                img[(x, y)] = v;
            }
        }
        let cfg = DetectorConfig {
            use_gpu: false,
            ..DetectorConfig::default()
        };
        let det = Detector::new(demo_face_cascade(), cfg);
        let plain = det.detect(&img);
        let timed = det.detect_timed(&img);
        assert_eq!(plain.len(), timed.detections.len());
        for (a, b) in plain.iter().zip(timed.detections.iter()) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.score.to_bits(), b.score.to_bits());
        }
        assert!(timed.levels >= 1);
        assert!(timed.windows_evaluated > 0);
    }
}
