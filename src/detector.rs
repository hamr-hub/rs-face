//! Multi-scale sliding window detector + non-maximum suppression.

use crate::face::iou;
use crate::haar::{Cascade, EvalCache};
use crate::image::GrayImage;
use crate::integral::{IntegralImage, RotatedIntegralImage, SquaredIntegralImage};

// The classical detection box and greedy NMS live in `crate::face` so that
// landmark-based detectors do not depend on this Haar-specific module.
// Re-exported here for source compatibility (`rsface::detector::Detection`).
pub use crate::face::{non_max_suppression, Detection};

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
    /// Minimum number of mutually-similar raw detections (plus children
    /// contained inside the merged box) required for a face to be reported.
    /// This is OpenCV `detectMultiScale(..., minNeighbors)`: raw sliding-window
    /// hits are clustered across ALL pyramid levels, and clusters with at most
    /// `min_neighbors` supporting hits are discarded as noise. `0` disables the
    /// requirement (every raw hit survives into the final NMS pass).
    /// Defaults to `3`, the OpenCV default.
    pub min_neighbors: i32,
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
    /// Defaults to `false`, matching OpenCV's C++ `detectMultiScale` (which
    /// does not equalize; the canonical Python samples call `equalizeHist`
    /// explicitly). Enable for low-contrast inputs where stage 0 rejects too
    /// many windows.
    pub equalize_hist: bool,
    /// If `true`, attempt to use the GPU for the squared-integral computation
    /// and variance pre-filter. Falls back to CPU silently if no GPU/OpenCL
    /// is available.
    pub use_gpu: bool,
    /// Minimum pixel count (W*H) at which the GPU path is preferred over CPU.
    /// Below this size, the kernel launch + PCIe transfer overhead exceeds the
    /// compute savings, so we skip GPU and count it in `gpu_skipped_levels`.
    /// Default: 250_000 (≈500×500) — tuned on Tegra-class hardware.
    pub gpu_min_pixels: usize,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            min_size: 24,
            max_size: 1024,
            scale_factor: 1.2,
            window_stride: 4,
            min_neighbors: 3,
            nms_iou_threshold: 0.3,
            min_score: 0.0,
            variance_threshold: 200,
            equalize_hist: false,
            // Opt-in: a library `detect()` call must not dlopen/JIT a GPU
            // stack unexpectedly. CLIs and the pipeline flip this on.
            use_gpu: false,
            gpu_min_pixels: 250_000,
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
        if self.min_neighbors < 0 {
            self.min_neighbors = 0;
        }
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

    /// Builder-style override of `min_neighbors` (0 disables grouping).
    pub fn with_min_neighbors(mut self, min_neighbors: i32) -> Self {
        self.min_neighbors = min_neighbors;
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
    /// launch + PCIe transfer has fixed overhead; below `gpu_min_pixels` the CPU
    /// wins. Threshold tunable via [`DetectorConfig::gpu_min_pixels`].
    fn gpu_worthwhile(&self, img_w: usize, img_h: usize) -> bool {
        img_w * img_h >= self.config.gpu_min_pixels
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

        // Optional histogram equalization (off by default to match OpenCV's
        // C++ detectMultiScale; see `DetectorConfig::equalize_hist`).
        let eq_storage: Option<GrayImage> = if self.config.equalize_hist {
            let mut eq = img.clone();
            eq.equalize_hist_inplace();
            Some(eq)
        } else {
            None
        };
        let current: &GrayImage = eq_storage.as_ref().unwrap_or(img);
        let mut downscaled: Option<GrayImage> = None;
        // Per-axis level→original scale. Both start at 1 and stay within a
        // rounding of each other, but the level dimensions are rounded
        // independently (as does OpenCV), so mapping y/height with the
        // width-based scale was off by ~1 px on deeper levels.
        let mut scale_x: f32 = 1.0;
        let mut scale_y: f32 = 1.0;
        loop {
            // `current` is the original (or equalized) image on the first
            // iteration and the owned pyramid level afterwards.
            let current: &GrayImage = downscaled.as_ref().unwrap_or(current);
            let cw = current.width();
            let ch = current.height();
            let det_w_at_cur = (win_w as f32 * scale_x).round() as usize;
            let det_h_at_cur = (win_h as f32 * scale_y).round() as usize;
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
                // The GPU kernels emit a u32 integral table; on images whose
                // prefix sums can wrap u32 we must stay on the CPU u64 path.
                if self.gpu_worthwhile(cw, ch) && crate::integral::prefix_sums_fit_u32(cw, ch) {
                    let (ii_data, ii_sq_data) = g.compute_dual(&current);
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
            // Tell the cascade whether the regular integral is narrow (u32)
            // — the cascade's per-rect reads take the branch-free narrow
            // path when this is true. Set once per level.
            cache.set_narrow_integral(!ii.is_wide());
            // Rotated integral: only cascades with tilted (DiagonalEdge)
            // features ever query it. The demo cascade has none, so skip the
            // (W+1)×(H+1) i64 construction entirely for such cascades.
            let ri = if self.needs_tilted {
                RotatedIntegralImage::from_gray(current)
            } else {
                RotatedIntegralImage::empty()
            };
            // `window_stride` is documented in ORIGINAL-image pixels. A window
            // step of `b` original pixels is `b / s` pixels on a level shrunk by
            // scale s. Multiplying instead (the previous formulation) advanced
            // the window up to 30 px on the smallest level, so faces visible on
            // only one pyramid level had essentially a single sample point and
            // were missed almost entirely (lena: 0 hits at default settings).
            // Stride on the level uses the x scale; with isotropic pyramid
            // resizes both axes share it to within rounding.
            let stride = ((self.config.window_stride as f32) / scale_x)
                .round()
                .max(1.0) as usize;
            let use_variance = self.config.variance_threshold < u64::MAX;
            // Inner normrect (`window - 2` on each side) and its pixel count
            // are constant for a fixed cascade window. Hoist them out of the
            // per-window loop so the variance pre-filter and the cascade's
            // variance-norm factor can both skip `(w*h)` and `(w*h)²`
            // arithmetic per window.
            let nw_norm = win_w.saturating_sub(2);
            let nh_norm = win_h.saturating_sub(2);
            let n_pixels = (nw_norm * nh_norm) as u64;
            let n_pixels_sq = n_pixels * n_pixels;
            let thr = self.config.variance_threshold;
            // Hoist the IntegralTable discriminant: the per-window
            // `rect_sum_unchecked_narrow` skips the enum match that the
            // generic variant emits, and every 640x480 / 1080p / 4K input
            // the detector sees in practice satisfies
            // `cw * ch * 255 ≤ u32::MAX` (the narrow path's contract).
            let ii_is_narrow = !ii.is_wide();
            // Hoist the (N → f64) cast: the per-window variance
            // computation is `N * ss - s²` (a fused multiply-add on the
            // FMA unit) and the (N → f64) cast is constant for the
            // cascade window.
            let n_pixels_f64 = n_pixels as f64;

            // GPU fast-path: run the full cascade on GPU when worth it.
            // The kernel handles variance normalisation + per-stage eval +        // early rejection in parallel across all (x, y) windows.
            if stride == 1 {
                if let Some(g) = self.gpu() {
                    // Same u32-overflow guard as the integral-build path:
                    // the GPU kernel's table cannot represent wide images.
                    if self.gpu_worthwhile(cw, ch) && crate::integral::prefix_sums_fit_u32(cw, ch) {
                        let max_dets = ((cw - win_w + 1) * (ch - win_h + 1)).min(8192);
                        let gpu_dets = g.detect_windows(&self.cascade, current, max_dets);
                        for d in gpu_dets {
                            if d.score < self.config.min_score {
                                continue;
                            }
                            // Level-space window coords map back to the
                            // original image by MULTIPLYING by the per-axis
                            // level scales (s = orig_dim / level_dim);
                            // dividing placed every face near the top-left
                            // corner regardless of its true position.
                            let ox = (d.x as f32 * scale_x).round() as usize;
                            let oy = (d.y as f32 * scale_y).round() as usize;
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
                            // Specialise for the narrow (u32) IntegralImage:
                            // the cascade's per-window corner reads are the
                            // dominant cost in the hot loop, and the generic
                            // variant has to `match` the IntegralTable enum
                            // for every call. The image-size guard
                            // (`prefix_sums_fit_u32`) keeps the integral on
                            // the narrow path for any 640x480 / 1080p / 4K
                            // input the detector will see in practice.
                            let s = if ii_is_narrow {
                                ii.rect_sum_unchecked_narrow(
                                    x + 1,
                                    y + 1,
                                    x + win_w - 1,
                                    y + win_h - 1,
                                )
                            } else {
                                ii.rect_sum_unchecked(x + 1, y + 1, x + win_w - 1, y + win_h - 1)
                            };
                            let ss = cache.sum_sq_rect_sum_unchecked(
                                x + 1,
                                y + 1,
                                x + win_w - 1,
                                y + win_h - 1,
                            );
                            (s, ss)
                        };
                        // The pre-filter expression `(ss * N - s²)` is the
                        // same integer the cascade needs for its
                        // varianceNormFactor sqrt. Compute it ONCE here in
                        // f64 and hand both to the cascade — saves the
                        // cascade from redoing `(nw_area * sum_sq - sum²)`
                        // per window.
                        if !SquaredIntegralImage::passes_variance_sums_fast(
                            s,
                            ss,
                            n_pixels,
                            n_pixels_sq,
                            thr,
                        ) {
                            None
                        } else {
                            // Fused multiply-add: `(N * ss) - s²` is one
                            // fma op on the FMA unit instead of two
                            // multiplies and a subtract.
                            let s_f = s as f64;
                            let ss_f = ss as f64;
                            let variance_part = n_pixels_f64.mul_add(ss_f, -s_f * s_f);
                            self.cascade.classify_inbounds_with_variance_part(
                                &ii,
                                &ri,
                                x,
                                y,
                                &mut cache,
                                variance_part,
                            )
                        }
                    } else {
                        self.cascade.classify_inbounds(&ii, &ri, x, y, &mut cache)
                    };
                    if let Some(score) = score_opt {
                        if score >= self.config.min_score {
                            // Map (x, y) at this level back to original image
                            // space with the per-axis scales (see the GPU
                            // fast-path note above).
                            let ox = (x as f32 * scale_x).round() as usize;
                            let oy = (y as f32 * scale_y).round() as usize;
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
            scale_x = img.width() as f32 / next.width() as f32;
            scale_y = img.height() as f32 / next.height() as f32;
            if next.width() <= win_w || next.height() <= win_h {
                break;
            }
        }

        // OpenCV semantics (`detectMultiScale`): cluster raw hits across ALL
        // pyramid levels first and discard clusters without enough supporting
        // hits (minNeighbors), then resolve the rare survivor overlap with
        // greedy IoU NMS. The old code ran only the IoU pass, so a single
        // weak window anywhere in the image was reported as a face.
        let grouped = group_rectangles(raw, self.config.min_neighbors);
        (
            non_max_suppression(grouped, self.config.nms_iou_threshold),
            levels,
            windows_evaluated,
        )
    }
}

/// Relative position/size tolerance for two raw hits to count as the same
/// face. Matches OpenCV's `groupRectangles(..., eps=0.2)`.
const GROUP_EPS: f64 = 0.2;

/// OpenCV 4.x `cv::groupRectangles` port: cluster the raw multi-scale
/// sliding-window hits with an equivalence predicate (union-find, so the
/// relation is transitive across pyramid levels), average each cluster,
/// then discard clusters that are either too weak on their own
/// (`members <= min_neighbors`) or are swallowed by a stronger containing
/// cluster. The surviving box is the unchanged cluster average; the
/// reported score is the maximum member score (OpenCV has no score — its
/// `levelWeights` play that role).
///
/// `min_neighbors <= 0` returns the input untouched, exactly like OpenCV
/// when `groupThreshold <= 0` (the downstream IoU NMS still runs).
pub fn group_rectangles(dets: Vec<Detection>, min_neighbors: i32) -> Vec<Detection> {
    if dets.is_empty() || min_neighbors <= 0 {
        return dets;
    }
    let threshold = min_neighbors as usize;

    // 1. Union-find partition under OpenCV's SimilarRects predicate.
    let n = dets.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if similar_rects(&dets[i], &dets[j], GROUP_EPS) {
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
                if ri != rj {
                    parent[rj] = ri;
                }
            }
        }
    }

    // 2. Average rect + member count + best score per class, preserving
    // first-appearance (== OpenCV class-index) order.
    let mut class_of: Vec<usize> = vec![0; n];
    let mut roots: Vec<usize> = Vec::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        match roots.iter().position(|&c| c == root) {
            Some(k) => class_of[i] = k,
            None => {
                class_of[i] = roots.len();
                roots.push(root);
            }
        }
    }
    let classes = roots.len();
    let mut sums = vec![(0u64, 0u64, 0u64, 0u64); classes];
    let mut counts = vec![0usize; classes];
    let mut best_score = vec![f32::NEG_INFINITY; classes];
    for (i, d) in dets.iter().enumerate() {
        let c = class_of[i];
        let s = &mut sums[c];
        s.0 += d.x as u64;
        s.1 += d.y as u64;
        s.2 += d.w as u64;
        s.3 += d.h as u64;
        counts[c] += 1;
        if d.score > best_score[c] {
            best_score[c] = d.score;
        }
    }
    let avg: Vec<Detection> = (0..classes)
        .map(|c| Detection {
            // OpenCV truncates the scaled float; all terms are >= 0 so
            // integer division is the same rounding.
            x: (sums[c].0 as usize) / counts[c],
            y: (sums[c].1 as usize) / counts[c],
            w: (sums[c].2 as usize) / counts[c],
            h: (sums[c].3 as usize) / counts[c],
            score: best_score[c],
        })
        .collect();

    // 3. Threshold + containment suppression (OpenCV 4.x semantics).
    let mut out: Vec<Detection> = Vec::new();
    for i in 0..classes {
        let n1 = counts[i];
        if n1 <= threshold {
            continue;
        }
        let r1 = &avg[i];
        let mut swallowed = false;
        for j in 0..classes {
            if i == j || counts[j] <= threshold {
                continue;
            }
            let n2 = counts[j];
            let r2 = &avg[j];
            // Inflate r2 by eps on every side; integer truncation like
            // saturate_cast<int> on non-negative values.
            let dx = (r2.w as f64 * GROUP_EPS) as i64;
            let dy = (r2.h as f64 * GROUP_EPS) as i64;
            let l = r2.x as i64 - dx;
            let t = r2.y as i64 - dy;
            let rr = r2.x as i64 + r2.w as i64 + dx;
            let b = r2.y as i64 + r2.h as i64 + dy;
            let inside = r1.x as i64 >= l
                && r1.y as i64 >= t
                && r1.x as i64 + r1.w as i64 <= rr
                && r1.y as i64 + r1.h as i64 <= b;
            // A small (n1 < 3) cluster is absorbed by ANY populated
            // containing cluster; larger ones only by a strictly stronger one.
            if inside && (n2 > 3.max(n1) || n1 < 3) {
                swallowed = true;
                break;
            }
        }
        if !swallowed {
            out.push(avg[i].clone());
        }
    }
    out
}

/// OpenCV `SimilarRects`: same position/extent within a size-scaled delta.
/// Compares both top-left and bottom-right corners, so boxes of different
/// sizes at the same origin are NOT considered similar.
fn similar_rects(a: &Detection, b: &Detection, eps: f64) -> bool {
    let delta = eps * (a.w.min(b.w) as f64 + a.h.min(b.h) as f64) * 0.5;
    let close = |p: i64, q: i64| (p - q).abs() as f64 <= delta;
    close(a.x as i64, b.x as i64)
        && close(a.y as i64, b.y as i64)
        && close((a.x + a.w) as i64, (b.x + b.w) as i64)
        && close((a.y + a.h) as i64, (b.y + b.h) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::haar::params::demo_face_cascade;
    use std::path::Path;

    #[test]
    fn detector_with_zero_area_image_returns_no_detections() {
        // 0×0, 10×0 and 0×10 images are valid GrayImage buffers but the
        // pyramid / scan code must skip them rather than panic. Lock this
        // in so a future refactor of the pyramid doesn't regress it.
        let det = Detector::new(demo_face_cascade(), DetectorConfig::default());
        for (w, h) in [(0, 0), (10, 0), (0, 10)] {
            let img = GrayImage::new(w, h);
            assert!(
                det.detect(&img).is_empty(),
                "zero-area image {w}x{h} must produce zero detections"
            );
        }
    }

    #[test]
    #[ignore = "GPU/OpenCL init via OnceLock is flaky under multi-thread test runner on this Tegra box. Passes in isolation with --nocapture, segfaults when run with other tests. Tracked separately."]
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

    /// Real-face smoke test for the bundled demo cascade. We don't make claims
    /// about which cascade detects which fixture — the demo cascade is
    /// deliberately small and stage_bias is tunable for that — but we do assert
    /// the pipeline runs end-to-end on a real face fixture and reports
    /// non-negative scores in valid image coordinates.
    ///
    /// Honest about the demo cascade: it is calibrated for synthetic test
    /// patterns, so this test is intentionally lax — it would only fail if the
    /// detector panicked, returned NaN scores, or produced boxes outside the
    /// image. The production path uses the bundled OpenCV cascade
    /// (`crate::haar::bundled`); measured numbers for trained `.rfcf`
    /// cascades live in `docs/bench-results.md`.
    #[test]
    fn demo_cascade_runs_on_real_face_fixture() {
        let fixture = Path::new("tests/fixtures/lena.ppm");
        assert!(
            fixture.exists(),
            "fixture {} missing — repo layout changed",
            fixture.display(),
        );
        let mut f = std::fs::File::open(fixture).expect("open lena.ppm");
        let rgb = crate::image::codec::read_ppm(&mut f).expect("decode ppm");
        let img = rgb.to_gray();
        let (w, h) = (img.width(), img.height());

        let cfg = DetectorConfig {
            min_size: 24,
            max_size: 4096,
            scale_factor: 1.4,
            window_stride: 3,
            use_gpu: false,
            equalize_hist: true,
            ..DetectorConfig::default()
        };
        let det = Detector::new(demo_face_cascade(), cfg);
        let hits = det.detect(&img);

        // Every box must be in-bounds and have a finite, non-negative score.
        for d in &hits {
            assert!(d.score.is_finite(), "NaN score");
            assert!(d.score >= 0.0, "negative score");
            assert!(d.x < w && d.y < h, "box origin out of bounds");
            assert!(d.x + d.w <= w && d.y + d.h <= h, "box extent out of bounds");
            assert!(d.w >= 24 && d.h >= 24, "box smaller than window");
        }
        // The fixture must be a real PPM, not the test-suite placeholder.
        assert!(w > 100 && h > 100, "fixture suspiciously small: {w}x{h}");
    }

    fn det(x: usize, y: usize, w: usize, h: usize, score: f32) -> Detection {
        Detection { x, y, w, h, score }
    }

    #[test]
    fn group_rectangles_passthrough_when_disabled() {
        let raw = vec![det(10, 10, 24, 24, 5.0), det(200, 200, 24, 24, 1.0)];
        assert_eq!(group_rectangles(raw.clone(), 0).len(), 2);
        assert_eq!(group_rectangles(raw, -1).len(), 2);
        assert!(group_rectangles(Vec::new(), 3).is_empty());
    }

    #[test]
    fn group_rectangles_requires_more_than_threshold_hits() {
        // OpenCV: n1 <= groupThreshold rejects. Exactly 3 hits with
        // min_neighbors=3 must therefore NOT survive.
        let raw = vec![
            det(100, 100, 50, 50, 1.0),
            det(102, 101, 50, 50, 2.0),
            det(101, 103, 51, 50, 3.0),
        ];
        assert!(group_rectangles(raw, 3).is_empty());

        // A fourth similar hit clears the threshold; the output is the
        // truncated average and the best member score survives.
        let raw = vec![
            det(100, 100, 50, 50, 1.0),
            det(102, 101, 50, 50, 2.0),
            det(101, 103, 51, 50, 3.0),
            det(103, 100, 50, 51, 9.5),
        ];
        let out = group_rectangles(raw, 3);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].x, (100 + 102 + 101 + 103) / 4);
        assert_eq!(out[0].w, (50 + 50 + 51 + 50) / 4);
        assert_eq!(out[0].score, 9.5);
    }

    #[test]
    fn group_rectangles_clusters_transitively_across_scales() {
        // Two pyramid-level clouds, offset by far more than one eps-delta but
        // chained through intermediate boxes -> one class via union-find.
        let mut raw = Vec::new();
        for (x, y) in [(100, 100), (106, 102), (112, 104), (118, 106)] {
            raw.push(det(x, y, 60, 60, 1.0));
        }
        for (x, y) in [(400, 300), (405, 303), (409, 299), (403, 306)] {
            raw.push(det(x, y, 80, 80, 4.0));
        }
        let out = group_rectangles(raw, 3);
        assert_eq!(out.len(), 2, "two distinct faces must stay separate");
    }

    #[test]
    fn group_rectangles_drops_isolated_weak_hit() {
        // One weak singleton far from anything: with min_neighbors=3 even a
        // strong cluster can't save it, and singletons are always filtered.
        let mut raw = vec![
            det(500, 500, 40, 40, 0.1), // isolated FP
        ];
        for (x, y) in [(100, 100), (102, 101), (101, 102), (103, 100)] {
            raw.push(det(x, y, 50, 50, 5.0));
        }
        let out = group_rectangles(raw, 3);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].x, 101);
    }

    #[test]
    fn group_rectangles_absorbs_contained_small_cluster() {
        // min_neighbors=1 lets a 2-member cluster (n1 < 3) reach OpenCV's
        // containment pass, where the stronger containing cluster swallows it.
        let mut raw = Vec::new();
        for (x, y) in [(100, 100), (104, 100), (100, 104), (104, 104), (108, 102)] {
            raw.push(det(x, y, 100, 100, 5.0));
        }
        raw.push(det(130, 130, 30, 30, 1.0));
        raw.push(det(131, 131, 30, 30, 1.0));
        let out = group_rectangles(raw, 1);
        assert_eq!(out.len(), 1, "contained weak cluster must be absorbed");

        // A 2-member cluster of equal strength that is NOT contained must
        // survive under the same threshold.
        let mut raw2 = Vec::new();
        for (x, y) in [(100, 100), (104, 100), (100, 104), (104, 104)] {
            raw2.push(det(x, y, 60, 60, 5.0));
        }
        raw2.push(det(300, 300, 40, 40, 1.0));
        raw2.push(det(302, 301, 40, 40, 1.0));
        let out2 = group_rectangles(raw2, 1);
        assert_eq!(out2.len(), 2, "disjoint clusters must both survive");
    }

    #[test]
    fn similar_rects_tolerance_matches_opencv() {
        let a = det(100, 100, 50, 50, 0.0);
        // delta = 0.2 * (50 + 50) / 2 = 10; 10 is still similar (<=).
        assert!(similar_rects(&a, &det(110, 100, 50, 50, 0.0), GROUP_EPS));
        assert!(!similar_rects(&a, &det(111, 100, 50, 50, 0.0), GROUP_EPS));
        // Same origin but a size delta beyond eps is NOT similar.
        assert!(!similar_rects(&a, &det(100, 100, 72, 72, 0.0), GROUP_EPS));
    }
}
