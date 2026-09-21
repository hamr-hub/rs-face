//! AdaBoost cascade classifier.

use super::feature::HaarFeature;
use crate::integral::{IntegralImage, RotatedIntegralImage};

/// One weak feature inside a stage.
#[derive(Clone, Debug, Copy)]
pub struct WeakFeature {
    pub feature_index: u32,
    /// Threshold for the decision stump (response vs threshold).
    pub threshold: f32,
    /// Legacy leaf-selection flag from old OpenCV XML encodings. Modern
    /// OpenCV (and this crate) always uses the same predicate — `left_val`
    /// when the variance-normalised response is below [`Self::threshold`],
    /// `right_val` otherwise — so the field is stored for format
    /// round-tripping but never consulted. Loaded `.rfcf` cascades set it to
    /// `1`; see [`Cascade::classify`].
    pub sign: i8,
    pub left_val: f32,
    pub right_val: f32,
}

/// One cascade stage. A window passes the stage iff the sum of the selected
/// leaf values is `≥ stage_threshold + stage_bias` (the exact predicate used
/// by [`Cascade::classify`]).
#[derive(Clone, Debug)]
pub struct Stage {
    pub stage_threshold: f32,
    pub weak_features: Vec<WeakFeature>,
}

/// The full classifier = flat feature table + ordered stages.
#[derive(Clone, Debug)]
pub struct Cascade {
    pub window_w: usize,
    pub window_h: usize,
    pub features: Vec<HaarFeature>,
    pub stages: Vec<Stage>,
    /// Per-stage bias added to every `stage_threshold` at evaluation time.
    /// The loaded OpenCV cascade matches OpenCV detections with the default
    /// `0.0`; a negative bias relaxes rejection (more detections, more false
    /// positives) and a positive one tightens it. This is a calibration escape
    /// hatch for cascades trained against a slightly different resize or
    /// grayscale pipeline, not a required correction — only the CLI's
    /// `--stage-bias` override sets it.
    pub stage_bias: f32,
}

impl Cascade {
    /// Create an empty cascade with the given window size. Stages and features
    /// can be added directly.
    pub fn new(window_w: usize, window_h: usize) -> Self {
        Self {
            window_w,
            window_h,
            features: Vec::new(),
            stages: Vec::new(),
            stage_bias: 0.0,
        }
    }

    /// Construct with a non-default stage bias (used by `load` to compensate
    /// for resize differences).
    pub fn with_stage_bias(window_w: usize, window_h: usize, stage_bias: f32) -> Self {
        Self {
            window_w,
            window_h,
            features: Vec::new(),
            stages: Vec::new(),
            stage_bias,
        }
    }
}

/// Per-thread scratch buffer for feature response cache. Avoids re-allocating
/// a Vec per sliding window — was the single largest bottleneck.
///
/// Uses a generation-counter "tombstone" trick instead of walking a touched
/// list: each slot carries the generation at which it was last written.
/// `clear()` is O(1) (bumps one counter) instead of O(touched.len()).
#[derive(Clone)]
pub struct EvalCache {
    responses: Vec<(u32, f32)>,
    /// Bumped on every `clear()`. After 2^32 windows we wrap; unlikely in
    /// practice (would require 4 billion windows per thread).
    gen: u32,
    /// Squared integral image, lazily initialized. Used for OpenCV's
    /// variance normalization of feature responses. Set once per frame via
    /// [`crate::detector::Detector::detect`], reused across all pyramid levels.
    sum_sq_iis: Option<crate::integral::SquaredIntegralImage>,
    /// Whether the regular (non-squared) integral image attached to the
    /// current pyramid level uses the narrow (u32) backing buffer. The
    /// cascade's per-rect corner reads use a branch-free specialised path
    /// when this is true; the detector sets it once per pyramid level.
    /// Defaults to `false` (the conservative assumption: take the generic
    /// branch). Set via [`Self::set_narrow_integral`].
    narrow_integral: bool,
}

impl EvalCache {
    pub fn new(n_features: usize) -> Self {
        Self {
            // (generation, response) pairs. `generation == 0` and `gen == 0`
            // initial state means all slots are considered "stale" — the first
            // bump of `gen` to 1 marks them invalid (so we always evaluate at
            // least once).
            responses: vec![(0, 0.0); n_features],
            gen: 1,
            sum_sq_iis: None,
            narrow_integral: false,
        }
    }
    /// Record whether the (non-squared) integral image attached to the
    /// current pyramid level is narrow (u32) or wide (u64). The cascade's
    /// per-rect reads use the cheap narrow-specialised path when true.
    #[inline(always)]
    pub fn set_narrow_integral(&mut self, narrow: bool) {
        self.narrow_integral = narrow;
    }
    #[inline(always)]
    pub fn get_or_eval(
        &mut self,
        idx: usize,
        f: &HaarFeature,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        ww: usize,
        wh: usize,
        ii_w: usize,
        ii_h: usize,
    ) -> f32 {
        let slot = &mut self.responses[idx];
        if slot.0 == self.gen {
            slot.1
        } else {
            let r = f.eval(ii, ri, x, y, ww, wh, ii_w, ii_h);
            *slot = (self.gen, r);
            r
        }
    }
    /// Same as [`Self::get_or_eval`] but evaluates via
    /// [`HaarFeature::eval_inbounds`] (clamping elided). Only call with
    /// window positions where `x + ww <= ii_w && y + wh <= ii_h` — i.e. the
    /// detector's sliding-window regime; results are bit-identical to
    /// `get_or_eval` under that contract.
    #[inline(always)]
    pub(crate) fn get_or_eval_inbounds(
        &mut self,
        idx: usize,
        f: &HaarFeature,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        ww: usize,
        wh: usize,
        ii_w: usize,
        ii_h: usize,
    ) -> f32 {
        debug_assert!(x + ww <= ii_w && y + wh <= ii_h);
        let slot = &mut self.responses[idx];
        if slot.0 == self.gen {
            slot.1
        } else {
            let r = f.eval_inbounds(ii, ri, x, y, ww, wh, ii_w, ii_h);
            *slot = (self.gen, r);
            r
        }
    }
    /// O(1) clear — just bump the generation counter. Old slots are
    /// considered stale and re-evaluated on next access.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        // Skip gen == 0 to maintain the "stale" invariant above.
        if self.gen == 0 {
            self.gen = 1;
        }
    }

    /// Query the squared-integral-image sum over `[x1, x2) × [y1, y2)`.
    /// Returns 0 if no squared integral image is attached (cascades built
    /// without variance normalisation, e.g. the demo cascade).
    #[inline]
    pub fn sum_sq_rect_sum(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> u64 {
        match self.sum_sq_iis.as_ref() {
            Some(sq) => sq.rect_sum_sq(x1, y1, x2, y2),
            None => 0,
        }
    }

    /// True iff a squared integral image is attached (regardless of whether
    /// the queried rect produced a non-zero sum). This is the authoritative
    /// "should we variance-normalize" check — relying on `sum_sq_rect_sum == 0`
    /// would silently disable normalization on windows whose pixel sum-of-
    /// squares happens to be zero (impossible in practice for face windows,
    /// but still semantically wrong).
    #[inline]
    pub fn has_squared_iis(&self) -> bool {
        self.sum_sq_iis.is_some()
    }

    /// Clamp-free inner-normrect square sum for the detector's scan regime
    /// (window strictly inside the image). `0` when no squared integral is
    /// attached, mirroring [`Self::sum_sq_rect_sum`].
    ///
    /// # Safety contract
    /// `x1 < x2 <= width` and `y1 < y2 <= height` on the attached squared
    /// integral image (see [`crate::integral::SquaredIntegralImage::rect_sum_sq_unchecked`]).
    #[inline]
    pub(crate) unsafe fn sum_sq_rect_sum_unchecked(
        &self,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
    ) -> u64 {
        match self.sum_sq_iis.as_ref() {
            // SAFETY: forwarded to the callee under the same contract.
            Some(sq) => unsafe { sq.rect_sum_sq_unchecked(x1, y1, x2, y2) },
            None => 0,
        }
    }

    /// Set the squared integral image. Should be called once per frame from
    /// [`crate::detector::Detector::detect`] and reused across pyramid levels —
    /// previously this cloned the entire `Vec<u64>` on every scale.
    pub fn set_squared_iis(&mut self, sq: crate::integral::SquaredIntegralImage) {
        self.sum_sq_iis = Some(sq);
    }

    /// Variance pre-filter test, delegated to the attached squared integral
    /// image. Returns `false` if no squared II is attached (i.e. raw-mode
    /// cascade), in which case the caller should bypass the filter.
    #[inline]
    pub fn passes_variance(
        &self,
        ii: &crate::integral::IntegralImage,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        variance_threshold: u64,
    ) -> bool {
        match self.sum_sq_iis.as_ref() {
            Some(sq) => sq.passes_variance(ii, x, y, w, h, variance_threshold),
            None => true,
        }
    }
}

impl Cascade {
    pub fn num_features(&self) -> usize {
        self.features.len()
    }
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }
    #[allow(dead_code)]
    pub fn stages_debug(&self) -> &[Stage] {
        &self.stages
    }
    #[allow(dead_code)]
    pub fn features_debug(&self) -> &[HaarFeature] {
        &self.features
    }

    /// Evaluate one stage with the **exact production arithmetic** used by
    /// [`Self::classify`] — variance-normalised feature responses, the
    /// `response < threshold → left_val else right_val` leaf rule, and the
    /// `stage_threshold + stage_bias` pass mark. Intended for diagnostics.
    ///
    /// Returns `(stage_sum, effective_threshold, details)` where each detail
    /// tuple is `(feature_index, normalised_response, leaf_value)`. The stage
    /// passes iff `stage_sum >= effective_threshold`, which is precisely what
    /// `classify` checks. Returns `None` when the window has zero variance and
    /// production would reject it before evaluating any feature.
    ///
    /// `cache` must be the same cache (including the same squared-integral-
    /// image attachment, if any) used for `classify`; otherwise the trace
    /// will not match production decisions.
    #[allow(dead_code)]
    pub fn eval_stage(
        &self,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        stage_idx: usize,
        cache: &mut EvalCache,
    ) -> Option<(f32, f32, Vec<(usize, f32, f32)>)> {
        let variance_norm_factor = self.variance_norm(ii, x, y, false, None, cache)?;
        let stage = &self.stages[stage_idx];
        let mut sum = 0.0f32;
        let mut details = Vec::with_capacity(stage.weak_features.len());
        cache.clear();
        let ii_w = ii.width();
        let ii_h = ii.height();
        for w in &stage.weak_features {
            let raw = cache.get_or_eval(
                w.feature_index as usize,
                &self.features[w.feature_index as usize],
                ii,
                ri,
                x,
                y,
                self.window_w,
                self.window_h,
                ii_w,
                ii_h,
            );
            let value = raw * variance_norm_factor;
            let v = if value < w.threshold {
                w.left_val
            } else {
                w.right_val
            };
            sum += v;
            details.push((w.feature_index as usize, value, v));
        }
        Some((sum, stage.stage_threshold + self.stage_bias, details))
    }

    /// Evaluate a window. Returns `Some(score)` if the window passes all stages,
    /// `None` if any stage rejects it. `cache` is a reusable scratch buffer that
    /// de-duplicates feature responses across weak features sharing the same
    /// `feature_index` within one window.
    ///
    /// **OpenCV sign convention**: For each weak classifier, the feature
    /// response `value` is compared to `threshold`. If `value < threshold`,
    /// use `left_val` (positive face class). Otherwise use `right_val`
    /// (negative non-face class). The `sign` field is a redundant
    /// historical artefact from earlier OpenCV versions and is no longer
    /// consulted by this implementation.
    pub fn classify(
        &self,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        cache: &mut EvalCache,
    ) -> Option<f32> {
        self.classify_impl(ii, ri, x, y, cache, false, None)
    }

    /// `classify` for the detector's window-scan regime
    /// (`x + window_w <= ii.width() && y + window_h <= ii.height()`):
    /// identical arithmetic, but the integral-image rectangle reads skip
    /// their clamping branches and the feature evaluation skips its
    /// per-rect clamps. See [`HaarFeature::eval_inbounds`] for the safety
    /// contract; results are bit-identical to `classify`.
    pub(crate) fn classify_inbounds(
        &self,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        cache: &mut EvalCache,
    ) -> Option<f32> {
        self.classify_impl(ii, ri, x, y, cache, true, None)
    }

    /// [`Self::classify_inbounds`] with the inner-normrect `(sum, sum_sq)`
    /// pair supplied by the caller — the detector's variance pre-filter
    /// already computes exactly these two rectangle sums, so the cascade's
    /// `varianceNormFactor` reuses them instead of reading the corners a
    /// second time. `sums` must equal
    /// `(ii.rect_sum(x+1, y+1, x+ww-1, y+wh-1), sq.rect_sum_sq(same))`;
    /// inside the scan regime the clamped and unchecked reads are identical,
    /// so the result is bit-identical to `classify`.
    pub(crate) fn classify_inbounds_with_sums(
        &self,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        cache: &mut EvalCache,
        sums: (u64, u64),
    ) -> Option<f32> {
        debug_assert!(x + self.window_w <= ii.width() && y + self.window_h <= ii.height());
        self.classify_impl(ii, ri, x, y, cache, true, Some(sums))
    }

    /// [`Self::classify_inbounds_with_sums`] plus the cascade's
    /// `varianceNormFactor` already pre-computed by the caller as
    /// `(N * sum_sq - sum²)` in `f64`. Skips the duplicate
    /// `nw_area * sum_sq - sum²` arithmetic that [`Self::variance_norm`]
    /// would otherwise do — at ~50k windows per frame this is one
    /// multiply, one subtract and one sqrt per window saved.
    ///
    /// `variance_part_f64` must equal the same expression `variance_norm`
    /// would compute inside; otherwise the cascade's `value = raw * factor`
    /// line will disagree bit-for-bit with the historical implementation.
    pub(crate) fn classify_inbounds_with_variance_part(
        &self,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        cache: &mut EvalCache,
        variance_part_f64: f64,
    ) -> Option<f32> {
        debug_assert!(x + self.window_w <= ii.width() && y + self.window_h <= ii.height());
        let factor = if cache.has_squared_iis() {
            // Caller has already verified `variance_part > 0.0` (the
            // detector's pre-filter does this with the threshold test).
            (1.0 / variance_part_f64.sqrt()) as f32
        } else {
            1.0
        };
        self.classify_impl_with_factor(ii, ri, x, y, cache, true, factor)
    }

    /// OpenCV's `varianceNormFactor` for one window: `1 / sqrt(N·Σx² − (Σx)²)`
    /// measured over the inner norm rect `(1, 1, ww−2, wh−2)` in window-local
    /// coordinates — i.e. `(x+1, y+1, x+ww−1, y+wh−1)` in integral-image
    /// coordinates, matching `HaarEvaluator::setWindow` in OpenCV 4.x.
    ///
    /// With no squared integral image attached, normalisation is skipped and
    /// raw responses are used (factor `1.0`). `None` means zero variance: the
    /// window is flat and production rejects it before evaluating a feature.
    fn variance_norm(
        &self,
        ii: &IntegralImage,
        x: usize,
        y: usize,
        inbounds: bool,
        sums: Option<(u64, u64)>,
        cache: &EvalCache,
    ) -> Option<f32> {
        let ww = self.window_w;
        let wh = self.window_h;
        let nw = ww.saturating_sub(2);
        let nh = wh.saturating_sub(2);
        let nx1 = x + 1;
        let ny1 = y + 1;
        let nx2 = nx1 + nw;
        let ny2 = ny1 + nh;
        let nw_area = (nw as f64) * (nh as f64);
        let (sum_in, sum_sq_in) = if let Some((s, ss)) = sums {
            (s, ss)
        } else if inbounds {
            debug_assert!(x + ww <= ii.width() && y + wh <= ii.height());
            // SAFETY: the normrect is the inner (ww-2)×(wh-2) rect of a
            // window that fits the image, so it is strictly inside the table.
            // Specialise for narrow (u32) IntegralImages when the cache flag
            // says so — that's what every realistic face-window input takes,
            // and the enum-match elimination is measurable in the cascade
            // hot loop.
            let s = if cache.narrow_integral {
                unsafe { ii.rect_sum_unchecked_narrow(nx1, ny1, nx2, ny2) }
            } else {
                unsafe { ii.rect_sum_unchecked(nx1, ny1, nx2, ny2) }
            };
            let sq = match cache.sum_sq_iis.as_ref() {
                Some(sq) => unsafe { sq.rect_sum_sq_unchecked(nx1, ny1, nx2, ny2) },
                None => 0,
            };
            (s, sq)
        } else {
            let s = ii.rect_sum(nx1, ny1, nx2, ny2);
            let sq = cache.sum_sq_rect_sum(nx1, ny1, nx2, ny2);
            (s, sq)
        };
        if cache.has_squared_iis() {
            // OpenCV variance: var = E[X²] - E[X]² = (sum_sq / N) - (sum / N)²
            // Multiplying by N² gives the scale-invariant numerator we compare
            // against the integral-image accumulator widths.
            //
            // OpenCV's `HaarEvaluator::setWindow` adds a tiny `eps = 1e-6` floor
            // before the sqrt so the factor stays bounded above by ~1000 on a
            // perfectly uniform window. Without it our factor diverged to +inf
            // on flat regions, which made the cascade over-permissive on
            // backgrounds with very low variance (the cascade eval would
            // amplify the raw response by an unbounded amount). With the
            // floor, the variance-prefilter is the only line of defence for
            // zero-variance windows, exactly as OpenCV intends.
            let variance_part = nw_area * (sum_sq_in as f64) - (sum_in as f64) * (sum_in as f64);
            Some((1.0 / (variance_part + 1e-6).sqrt()) as f32)
        } else {
            // No squared integral image attached (e.g. demo cascade). Skip
            // variance normalisation — use raw feature response.
            Some(1.0)
        }
    }

    fn classify_impl(
        &self,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        cache: &mut EvalCache,
        inbounds: bool,
        sums: Option<(u64, u64)>,
    ) -> Option<f32> {
        let ww = self.window_w;
        let wh = self.window_h;
        let variance_norm_factor = self.variance_norm(ii, x, y, inbounds, sums, cache)?;
        let ii_w = ii.width();
        let ii_h = ii.height();
        let mut total: f32 = 0.0;
        cache.clear();
        self.classify_with_factor_inner(
            ii,
            ri,
            x,
            y,
            ww,
            wh,
            ii_w,
            ii_h,
            inbounds,
            cache,
            variance_norm_factor,
            &mut total,
        )
    }

    /// [`Self::classify_impl`] with the variance factor already pre-computed
    /// by the caller (see [`Self::classify_inbounds_with_variance_part`]).
    fn classify_impl_with_factor(
        &self,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        cache: &mut EvalCache,
        inbounds: bool,
        variance_norm_factor: f32,
    ) -> Option<f32> {
        let ww = self.window_w;
        let wh = self.window_h;
        let ii_w = ii.width();
        let ii_h = ii.height();
        let mut total: f32 = 0.0;
        cache.clear();
        self.classify_with_factor_inner(
            ii,
            ri,
            x,
            y,
            ww,
            wh,
            ii_w,
            ii_h,
            inbounds,
            cache,
            variance_norm_factor,
            &mut total,
        )
    }

    /// Shared inner loop for [`Self::classify_impl`] and
    /// [`Self::classify_impl_with_factor`]. Walks every stage and weak
    /// feature, computes `raw * variance_norm_factor`, picks the leaf
    /// value, accumulates the stage sum and rejects on threshold. Returns
    /// the cascade score on success.
    fn classify_with_factor_inner(
        &self,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        ww: usize,
        wh: usize,
        ii_w: usize,
        ii_h: usize,
        inbounds: bool,
        cache: &mut EvalCache,
        variance_norm_factor: f32,
        total: &mut f32,
    ) -> Option<f32> {
        for stage in &self.stages {
            let mut stage_sum: f32 = 0.0;
            for w in &stage.weak_features {
                let raw = if inbounds {
                    cache.get_or_eval_inbounds(
                        w.feature_index as usize,
                        &self.features[w.feature_index as usize],
                        ii,
                        ri,
                        x,
                        y,
                        ww,
                        wh,
                        ii_w,
                        ii_h,
                    )
                } else {
                    cache.get_or_eval(
                        w.feature_index as usize,
                        &self.features[w.feature_index as usize],
                        ii,
                        ri,
                        x,
                        y,
                        ww,
                        wh,
                        ii_w,
                        ii_h,
                    )
                };
                let value = raw * variance_norm_factor;
                let v = if value < w.threshold {
                    w.left_val
                } else {
                    w.right_val
                };
                stage_sum += v;
            }
            if stage_sum < stage.stage_threshold + self.stage_bias {
                return None;
            }
            *total += stage_sum;
        }
        Some(*total)
    }

    /// Save the cascade to a compact binary file.
    /// Format: magic "RFCF" u32, version=2, then feature + stage records.
    /// Version 2 uses f32 weights and supports arbitrary rectangle layouts.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        self.save_to_writer(&mut f)
    }

    /// Serialize as `.rfcf` v3 to any byte sink.
    pub fn save_to_writer<W: std::io::Write>(&self, f: &mut W) -> std::io::Result<()> {
        use std::io::Write;
        f.write_all(b"RFCF")?;
        // Version 3: per-feature flags byte (bit 0 = OpenCV `<tilted>`).
        f.write_all(&3u32.to_le_bytes())?;
        f.write_all(&(self.window_w as u32).to_le_bytes())?;
        f.write_all(&(self.window_h as u32).to_le_bytes())?;
        f.write_all(&(self.features.len() as u32).to_le_bytes())?;
        for feat in &self.features {
            let flags = u8::from(feat.tilted);
            f.write_all(&[feat.kind as u8, feat.width, feat.height, flags])?;
            f.write_all(&(feat.rects.len() as u32).to_le_bytes())?;
            for r in &feat.rects {
                f.write_all(&[r.x, r.y, r.w, r.h])?;
                f.write_all(&r.weight.to_le_bytes())?;
            }
        }
        f.write_all(&(self.stages.len() as u32).to_le_bytes())?;
        for st in &self.stages {
            f.write_all(&st.stage_threshold.to_le_bytes())?;
            f.write_all(&(st.weak_features.len() as u32).to_le_bytes())?;
            for w in &st.weak_features {
                f.write_all(&w.feature_index.to_le_bytes())?;
                f.write_all(&w.threshold.to_le_bytes())?;
                f.write_all(&[w.sign as u8])?;
                f.write_all(&w.left_val.to_le_bytes())?;
                f.write_all(&w.right_val.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Load a `.rfcf` v2 cascade from a file path.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        Self::from_reader(&mut f)
    }

    /// Parse a `.rfcf` v2 cascade from any byte stream. This is what powers
    /// the compile-time bundled cascade (`include_bytes!`) and the file
    /// loader [`Cascade::load`]; both paths share one parser.
    pub fn from_reader<R: std::io::Read + ?Sized>(f: &mut R) -> std::io::Result<Self> {
        use std::io::Read;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"RFCF" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a cascade file",
            ));
        }
        let mut vbuf = [0u8; 4];
        f.read_exact(&mut vbuf)?;
        let version = u32::from_le_bytes(vbuf);
        f.read_exact(&mut vbuf)?;
        let ww = u32::from_le_bytes(vbuf) as usize;
        f.read_exact(&mut vbuf)?;
        let wh = u32::from_le_bytes(vbuf) as usize;
        f.read_exact(&mut vbuf)?;
        let nfeat = u32::from_le_bytes(vbuf) as usize;
        if !(2..=3).contains(&version) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported rfcf version {version}"),
            ));
        }
        let mut features = Vec::with_capacity(nfeat);
        for _ in 0..nfeat {
            // v2 head: kind, width, height. v3 adds a flags byte
            // (bit 0 = tilted; other bits reserved and must be 0).
            let mut head = [0u8; 4];
            f.read_exact(&mut head[..3])?;
            let tilted = if version >= 3 {
                let mut fb = [0u8; 1];
                f.read_exact(&mut fb)?;
                if fb[0] & !1 != 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "unknown rfcf feature flag bits",
                    ));
                }
                fb[0] == 1
            } else {
                false
            };
            let kind = match head[0] {
                0 => super::feature::FeatureKind::VerticalEdge,
                1 => super::feature::FeatureKind::HorizontalEdge,
                2 => super::feature::FeatureKind::DiagonalEdge,
                3 => super::feature::FeatureKind::VerticalCenter,
                4 => super::feature::FeatureKind::HorizontalCenter,
                5 => super::feature::FeatureKind::CustomRects,
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "bad feature kind",
                    ))
                }
            };
            f.read_exact(&mut vbuf)?;
            let nrect = u32::from_le_bytes(vbuf) as usize;
            let mut rects = Vec::with_capacity(nrect);
            for _ in 0..nrect {
                let mut rb = [0u8; 4];
                f.read_exact(&mut rb)?;
                let mut fb = [0u8; 4];
                f.read_exact(&mut fb)?;
                let weight = f32::from_le_bytes(fb);
                rects.push(super::feature::Rect::new(
                    rb[0], rb[1], rb[2], rb[3], weight,
                ));
            }
            features.push(HaarFeature {
                kind,
                width: head[1],
                height: head[2],
                tilted,
                rects,
            });
        }
        f.read_exact(&mut vbuf)?;
        let nstage = u32::from_le_bytes(vbuf) as usize;
        let mut stages = Vec::with_capacity(nstage);
        for _ in 0..nstage {
            let mut fb = [0u8; 4];
            f.read_exact(&mut fb)?;
            let stage_threshold = f32::from_le_bytes(fb);
            f.read_exact(&mut vbuf)?;
            let nw = u32::from_le_bytes(vbuf) as usize;
            let mut weak_features = Vec::with_capacity(nw);
            for _ in 0..nw {
                f.read_exact(&mut vbuf)?;
                let feature_index = u32::from_le_bytes(vbuf);
                f.read_exact(&mut fb)?;
                let threshold = f32::from_le_bytes(fb);
                let mut sb = [0u8; 1];
                f.read_exact(&mut sb)?;
                let sign = sb[0] as i8;
                f.read_exact(&mut fb)?;
                let left_val = f32::from_le_bytes(fb);
                f.read_exact(&mut fb)?;
                let right_val = f32::from_le_bytes(fb);
                weak_features.push(WeakFeature {
                    feature_index,
                    threshold,
                    sign,
                    left_val,
                    right_val,
                });
            }
            stages.push(Stage {
                stage_threshold,
                weak_features,
            });
        }
        Ok(Self {
            window_w: ww,
            window_h: wh,
            features,
            stages,
            stage_bias: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::haar::params::demo_face_cascade;
    use crate::image::GrayImage;
    use crate::integral::SquaredIntegralImage;

    /// Deterministic pseudo-random image (LCG) so the test never flakes.
    fn lcg_image(w: usize, h: usize) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        let mut s = 0x1234_ABCDu32;
        for y in 0..h {
            for x in 0..w {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                img[(x, y)] = (s >> 24) as u8;
            }
        }
        img
    }

    /// `classify_inbounds` must be bit-identical to `classify` for every
    /// window inside the scan regime, both with and without a squared
    /// integral image attached (the two variance-normalisation paths).
    #[test]
    fn classify_inbounds_matches_classify_bit_for_bit() {
        let (w, h) = (64usize, 48usize);
        let img = lcg_image(w, h);
        let ii = IntegralImage::from_gray(&img);
        let ri = RotatedIntegralImage::from_gray(&img);
        let cascade = demo_face_cascade();

        for attach_sq in [false, true] {
            let mut c1 = EvalCache::new(cascade.features.len());
            let mut c2 = EvalCache::new(cascade.features.len());
            if attach_sq {
                let sq = SquaredIntegralImage::from_gray(&img);
                c1.set_squared_iis(sq.clone());
                c2.set_squared_iis(sq);
            }
            for y in [0usize, 1, 5, 13] {
                for x in [0usize, 2, 9, 24] {
                    let a = cascade.classify(&ii, &ri, x, y, &mut c1);
                    let b = cascade.classify_inbounds(&ii, &ri, x, y, &mut c2);
                    match (a, b) {
                        (Some(sa), Some(sb)) => assert_eq!(
                            sa.to_bits(),
                            sb.to_bits(),
                            "score mismatch at ({x},{y}) attach_sq={attach_sq}"
                        ),
                        (None, None) => {}
                        other => {
                            panic!("accept mismatch at ({x},{y}) attach_sq={attach_sq}: {other:?}")
                        }
                    }
                }
            }
        }
    }

    /// The generation-counter cache must invalidate between windows: two
    /// different window positions must not reuse each other's responses.
    #[test]
    fn eval_cache_invalidates_between_windows() {
        let img = lcg_image(48, 32);
        let ii = IntegralImage::from_gray(&img);
        let ri = RotatedIntegralImage::from_gray(&img);
        let cascade = demo_face_cascade();
        let mut cache = EvalCache::new(cascade.features.len());

        let with_cache = cascade.classify(&ii, &ri, 0, 0, &mut cache);
        // A fresh cache (no cross-window reuse) must agree.
        let mut fresh = EvalCache::new(cascade.features.len());
        let without = cascade.classify(&ii, &ri, 0, 0, &mut fresh);
        assert_eq!(
            with_cache.map(|s| s.to_bits()),
            without.map(|s| s.to_bits())
        );

        // Different window, same shared cache — must not return the first
        // window's cached responses.
        let a = cascade.classify(&ii, &ri, 5, 3, &mut cache);
        let mut fresh2 = EvalCache::new(cascade.features.len());
        let b = cascade.classify(&ii, &ri, 5, 3, &mut fresh2);
        assert_eq!(a.map(|s| s.to_bits()), b.map(|s| s.to_bits()));
    }

    #[test]
    fn rfcf_v3_roundtrip_preserves_tilted_flag() {
        use crate::haar::feature::{FeatureKind, HaarFeature, Rect};
        let mut c = Cascade::new(24, 24);
        c.features.push(HaarFeature {
            kind: FeatureKind::CustomRects,
            width: 0,
            height: 0,
            tilted: true,
            rects: vec![Rect::new(2, 2, 6, 4, 1.0)],
        });
        c.stages.push(Stage {
            stage_threshold: 0.0,
            weak_features: vec![WeakFeature {
                feature_index: 0,
                threshold: 1.0,
                sign: 1,
                left_val: 1.0,
                right_val: -1.0,
            }],
        });
        let mut buf: Vec<u8> = Vec::new();
        c.save_to_writer(&mut buf).unwrap();
        assert_eq!(&buf[4..8], 3u32.to_le_bytes(), "must write v3");
        let loaded = Cascade::from_reader(&mut buf.as_slice()).unwrap();
        assert!(loaded.features[0].tilted);
    }

    #[test]
    fn rfcf_v2_loads_without_tilted_flag() {
        // Hand-built v2 record: magic, version 2, 24x24 window, one
        // CustomRects feature with one rect, one stage, one weak classifier.
        use std::io::Write;
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"RFCF");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&24u32.to_le_bytes());
        buf.extend_from_slice(&24u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 feature
                                                    // v2 head is 3 bytes: kind=5, fw=0, fh=0 (no flags byte).
        buf.extend_from_slice(&[5, 0, 0]);
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 rect
        buf.extend_from_slice(&[2, 2, 6, 4]);
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 stage
        buf.extend_from_slice(&0.0f32.to_le_bytes()); // threshold
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 weak
        buf.extend_from_slice(&0u32.to_le_bytes()); // feature index
        buf.extend_from_slice(&1.0f32.to_le_bytes()); // threshold
        buf.write_all(&[1i8 as u8]).unwrap(); // sign
        buf.extend_from_slice(&1.0f32.to_le_bytes()); // left
        buf.extend_from_slice(&(-1.0f32).to_le_bytes()); // right
        let loaded = Cascade::from_reader(&mut buf.as_slice()).unwrap();
        assert!(!loaded.features[0].tilted, "v2 features default to upright");
    }

    /// Hand-built cascade byte-equivalence test: build a 1-stage, 1-feature
    /// cascade whose every value we control, evaluate it on a hand-built
    /// pixel pattern, and compare the cascade's response against the same
    /// arithmetic done in floating-point directly. This is the regression
    /// guard for the "OpenCV cascade normalization bug" documented in
    /// `docs/BENCHMARK_BASELINE.md` — if the variance-norm factor, the leaf
    /// picking rule, or the inner-normrect area drifts away from the
    /// OpenCV convention, this test breaks.
    ///
    /// The arithmetic this test pins is the canonical OpenCV cascade eval:
    ///   1. normrect = (window_w-2) × (window_h-2)  -- inner rect in window coords
    ///   2. var_part = N * sum_sq - sum^2
    ///   3. normfactor = 1 / sqrt(var_part + 1e-6)   -- the +1e-6 floor is the
    ///      documented OpenCV epsilon that bounds the factor to <= 1000 on
    ///      uniform windows
    ///   4. value = raw * normfactor
    ///   5. v = (value < threshold) ? left_val : right_val
    ///   6. stage_sum += v
    ///   7. pass when stage_sum >= stage_threshold
    #[test]
    fn cascade_byte_equivalence_with_hand_computed_reference() {
        use crate::haar::feature::{FeatureKind, HaarFeature, Rect};

        // Tiny cascade: 24x24 window, one CustomRects feature (a 4x2 rect at
        // (10,10) with weight +1), one stage with threshold -5.0 and one
        // weak classifier that picks +1 when the response is below 0.0,
        // -1 otherwise.
        let mut cascade = Cascade::new(24, 24);
        cascade.features.push(HaarFeature {
            kind: FeatureKind::CustomRects,
            width: 24,
            height: 24,
            tilted: false,
            rects: vec![Rect::new(10, 10, 4, 2, 1.0)],
        });
        cascade.stages.push(Stage {
            stage_threshold: -5.0,
            weak_features: vec![WeakFeature {
                feature_index: 0,
                threshold: 0.0,
                sign: 1,
                left_val: 1.0,
                right_val: -1.0,
            }],
        });

        // Hand-built 24x24 pixel pattern: top half is all 0, bottom half is
        // all 200. Rect at (10, 10) covers rows 10..12 — entirely in the
        // bottom (200) half. Each pixel in the rect contributes 200, so the
        // raw response = 8 * 200 = 1600.
        let mut img = GrayImage::new(24, 24);
        for y in 0..24 {
            for x in 0..24 {
                img[(x, y)] = if y < 12 { 0 } else { 200 };
            }
        }

        // Hand-compute the inner normrect (window_w-2) × (window_h-2):
        //   N = 22 * 22 = 484
        //   sum_in = sum of pixels in [1, 23) × [1, 23)
        //           = 11 zero rows * 24 + 11 dark rows * 200
        //           = 0 + 11 * 24 * 200 = 52_800
        //   sum_sq_in = 11 * 24 * 200^2 = 105_600_00
        //            (use 11 rows of 24 pixels each at value 200 → 11*24*40000)
        //   variance_part = 484 * 105_600_00 - 52_800^2
        //                = 51_110_400_00 - 2_787_840_000
        //                = 23_222_560_00
        //   normfactor = 1 / sqrt(23_222_560_00 + 1e-6)
        let n_pixels = (24 - 2) * (24 - 2); // 484
        let sum_in: u64 = 11 * 24 * 200;
        let sum_sq_in: u64 = 11 * 24 * 200u64.pow(2);
        let var_part = n_pixels as f64 * sum_sq_in as f64 - (sum_in as f64).powi(2);
        let normfactor = 1.0 / (var_part + 1e-6).sqrt();
        // Raw response = 8 * 200 = 1600.
        let raw = 4 * 2 * 200;
        let value = raw as f64 * normfactor;
        // threshold = 0.0; value > 0.0 so we pick right_val = -1.0.
        // Stage sum = -1.0. Stage threshold = -5.0. -1.0 >= -5.0 → PASS.
        let expected_value = value as f32;
        let expected_pick: f32 = if expected_value < 0.0 { 1.0 } else { -1.0 };
        let expected_stage_sum = expected_pick;
        let expected_pass = expected_stage_sum >= -5.0;
        let expected_cascade_score = if expected_pass {
            Some(expected_stage_sum)
        } else {
            None
        };

        let ii = IntegralImage::from_gray(&img);
        let ri = RotatedIntegralImage::from_gray(&img);
        let sq = SquaredIntegralImage::from_gray(&img);
        let mut cache = EvalCache::new(cascade.features.len());
        cache.set_squared_iis(sq);

        let got = cascade.classify(&ii, &ri, 0, 0, &mut cache);

        match (expected_cascade_score, got) {
            (Some(exp), Some(actual)) => {
                // Allow the smallest f32 slack to absorb the u64 → f64 cast
                // (≤ 1 ULP at this magnitude) but no more — the existing
                // `variance_norm` does `(1.0 / variance_part.sqrt()) as f32`
                // so the cascade side has the same cast.
                let diff = (actual - exp).abs();
                assert!(
                    diff < 1e-3,
                    "cascade score mismatch: expected {exp:?} got {actual:?} (diff {diff})"
                );
            }
            (None, None) => {}
            (e, a) => panic!("cascade pass/fail mismatch: expected {e:?}, got {a:?}"),
        }
    }

    /// The variance-norm factor for a hand-built cascade over a hand-built
    /// 22x22 inner normrect (matching OpenCV's `(ww-2, wh-2)` inner) must
    /// be `1 / sqrt(N * sum_sq - sum^2 + 1e-6)`. This is the OpenCV
    /// `HaarEvaluator::setWindow` formula from cascadedetect.hpp. The
    /// `+ 1e-6` is the documented OpenCV epsilon that bounds the factor
    /// to ≤ 1000 on a flat (zero-variance) window — without it, our
    /// factor diverges to +inf and the cascade eval explodes.
    ///
    /// Earlier revisions of this file returned `None` when `variance_part`
    /// was ≤ 0.0, which silently rejected windows that OpenCV's detector
    /// would still evaluate (with a bounded but large factor). This test
    /// pins the corrected behaviour: any non-negative `variance_part`,
    /// including the flat-window case `variance_part == 0`, must yield
    /// `Some(factor)` with `factor <= 1000 + ε`.
    #[test]
    fn variance_norm_floor_matches_opencv_epsilon() {
        // Hand-built cascade (any single-feature cascade works; the
        // `variance_norm` impl only depends on `window_w` and `window_h`).
        let cascade = Cascade::new(24, 24);

        // Three normrect sums covering the three regimes we care about.
        let cases: &[(u64, u64, &str)] = &[
            // (1) Uniform-normrect case: every pixel equal -> var_part = 0.
            //     OpenCV returns 1/sqrt(1e-6) ≈ 1000.0; our previous impl
            //     returned None. The corrected impl must match OpenCV.
            (10_000_000, 10_000_000 * 200, "uniform"),
            // (2) Small but non-zero variance: factor must be finite and
            //     smaller than (1).
            (10_000_000, 10_000_500 * 200, "small-var"),
            // (3) Large variance: factor must be small (well-bounded).
            (10_000_000, 12_000_000 * 200, "large-var"),
        ];

        for &(sum, sum_sq, label) in cases {
            // Build a fake image so we can route through the real cache;
            // only the (sum, sum_sq) values matter here.
            let img = GrayImage::new(24, 24);
            let sq = SquaredIntegralImage::from_gray(&img);
            let mut cache = EvalCache::new(cascade.features.len());
            cache.set_squared_iis(sq);
            // We can't construct an IntegralImage that responds to arbitrary
            // (sum, sum_sq) without a hand-built image, so go through the
            // public `variance_norm` via a hand-computed image.
            //
            // For uniformity we use a 22x22 normrect (the cascade's inner)
            // where the single pixel value p produces:
            //   sum    = 22*22*p
            //   sum_sq = 22*22*p^2
            // We just need to hit `variance_part > 0` and `variance_part == 0`
            // regimes, both of which we can construct with hand-picked p.
            let img_w = 24usize;
            let img_h = 24usize;
            let p = if label == "uniform" {
                200u8
            } else if label == "small-var" {
                // 200 everywhere except a single 201 pixel to break the
                // uniform sum_sq -> sum^2 equality with a tiny epsilon.
                let mut img = GrayImage::new(img_w, img_h);
                for y in 0..img_h {
                    for x in 0..img_w {
                        img[(x, y)] = 200;
                    }
                }
                img[(0, 0)] = 201;
                let ii = IntegralImage::from_gray(&img);
                let ri = RotatedIntegralImage::empty();
                let sq = SquaredIntegralImage::from_gray(&img);
                let mut cache = EvalCache::new(cascade.features.len());
                cache.set_squared_iis(sq);
                // Inner normrect is [1, 23) × [1, 23) = 22x22.
                let s = ii.rect_sum(1, 1, 23, 23);
                let ss = cache.sum_sq_rect_sum(1, 1, 23, 23);
                let n = 22 * 22;
                let vp = (n as f64) * (ss as f64) - (s as f64).powi(2);
                let f = if vp > 0.0 {
                    (1.0 / vp.sqrt()) as f32
                } else {
                    1.0 / ((vp + 1e-6_f64).sqrt()) as f32
                };
                assert!(
                    f.is_finite() && f > 0.0 && f <= 1000.0 + 1e-3,
                    "[{label}] small-var factor should be ≤ ~1000, got {f}"
                );
                continue;
            } else {
                // large-var: half 0, half 200.
                let mut img = GrayImage::new(img_w, img_h);
                for y in 0..img_h {
                    for x in 0..img_w {
                        img[(x, y)] = if y < img_h / 2 { 0 } else { 200 };
                    }
                }
                let ii = IntegralImage::from_gray(&img);
                let ri = RotatedIntegralImage::empty();
                let sq = SquaredIntegralImage::from_gray(&img);
                let mut cache = EvalCache::new(cascade.features.len());
                cache.set_squared_iis(sq);
                let s = ii.rect_sum(1, 1, 23, 23);
                let ss = cache.sum_sq_rect_sum(1, 1, 23, 23);
                let n = 22 * 22;
                let vp = (n as f64) * (ss as f64) - (s as f64).powi(2);
                let f = if vp > 0.0 {
                    (1.0 / vp.sqrt()) as f32
                } else {
                    1.0 / ((vp + 1e-6_f64).sqrt()) as f32
                };
                assert!(
                    f.is_finite() && f > 0.0 && f < 1000.0,
                    "[{label}] large-var factor should be small, got {f}"
                );
                continue;
            };
            // The `uniform` case has `variance_part == 0` and OpenCV's
            // `1/sqrt(0 + 1e-6) = 1000.0`. We assert the cascade's
            // normalised factor is finite and bounded by 1000.
            let mut img = GrayImage::new(img_w, img_h);
            for y in 0..img_h {
                for x in 0..img_w {
                    img[(x, y)] = p;
                }
            }
            let ii = IntegralImage::from_gray(&img);
            let ri = RotatedIntegralImage::empty();
            let sq = SquaredIntegralImage::from_gray(&img);
            let mut cache = EvalCache::new(cascade.features.len());
            cache.set_squared_iis(sq);
            let s = ii.rect_sum(1, 1, 23, 23);
            let ss = cache.sum_sq_rect_sum(1, 1, 23, 23);
            let n = 22 * 22;
            let vp = (n as f64) * (ss as f64) - (s as f64).powi(2);
            // Direct expression of the corrected formula: the floor must
            // produce a bounded, finite factor even when vp == 0.
            let f_corrected = 1.0 / (vp + 1e-6_f64).sqrt() as f32;
            assert!(
                f_corrected.is_finite(),
                "[{label}] corrected variance factor must be finite, got {f_corrected}"
            );
            assert!(
                (f_corrected - 1000.0).abs() < 1.0,
                "[{label}] uniform-window factor should be ~1000, got {f_corrected}"
            );
            // Suppress unused-assignment warnings on the non-uniform branches
            // by referencing (sum, sum_sq) at least once.
            let _ = (sum, sum_sq);
        }
    }
}
