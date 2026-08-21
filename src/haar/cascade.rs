//! AdaBoost cascade classifier.

use super::feature::HaarFeature;
use crate::integral::{IntegralImage, RotatedIntegralImage};

/// One weak feature inside a stage.
#[derive(Clone, Debug, Copy)]
pub struct WeakFeature {
    pub feature_index: u32,
    /// Threshold for the decision stump (response vs threshold).
    pub threshold: f32,
    /// Sign: `left_val` is used when response ≤ threshold (sign = -1)
    ///       or when response > threshold (sign = +1).
    pub sign: i8,
    pub left_val: f32,
    pub right_val: f32,
}

/// One cascade stage. A window passes the stage iff the weighted sum of weak
/// features (using `feature_value = if sign>0 { left_val if r≤t else right_val }`)
/// is `≥ stage_threshold`.
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
    /// Per-stage bias added to each `stage_threshold` after load. OpenCV's
    /// INTER_AREA resize produces slightly different integral sums than our
    /// `resize_area`; on real photographs the cascade needs ~-10 to match
    /// OpenCV's detection rate. Set to 0 if your cascade was trained against
    /// our exact pipeline.
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
    /// [`Detector::detect`], reused across all pyramid levels.
    sum_sq_iis: Option<crate::integral::SquaredIntegralImage>,
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
        }
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
    /// integral image (see [`SquaredIntegralImage::rect_sum_sq_unchecked`]).
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

    /// Evaluate one stage, returning the per-weak response and the sum.
    /// Used for diagnostics.
    #[allow(dead_code)]
    pub fn eval_stage(
        &self,
        ii: &IntegralImage,
        ri: &RotatedIntegralImage,
        x: usize,
        y: usize,
        stage_idx: usize,
    ) -> Option<(f32, Vec<(usize, f32, f32)>)> {
        let stage = &self.stages[stage_idx];
        let mut sum = 0.0f32;
        let mut details = Vec::new();
        for w in &stage.weak_features {
            let f = &self.features[w.feature_index as usize];
            let r = f.eval(
                ii,
                ri,
                x,
                y,
                self.window_w,
                self.window_h,
                ii.width(),
                ii.height(),
            );
            let v = if w.sign > 0 {
                if r > w.threshold {
                    w.right_val
                } else {
                    w.left_val
                }
            } else {
                if r > w.threshold {
                    w.left_val
                } else {
                    w.right_val
                }
            };
            sum += v;
            details.push((w.feature_index as usize, r, v));
        }
        Some((sum, details))
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
        // OpenCV's variance normalization: compute over the inner rect
        // (1, 1, ww-2, wh-2) in *window-local* coordinates, which means
        // (x+1, y+1, x+ww-1, y+wh-1) in integral-image coordinates.
        // Matches `HaarEvaluator::setWindow` in OpenCV 4.x.
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
            let s = unsafe { ii.rect_sum_unchecked(nx1, ny1, nx2, ny2) };
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
        let variance_norm_factor: f32 = if cache.has_squared_iis() {
            // OpenCV variance: var = E[X²] - E[X]² = (sum_sq / N) - (sum / N)²
            // Multiplying by N² gives the scale-invariant numerator we compare
            // against the integral-image accumulator widths.
            let variance_part = nw_area * (sum_sq_in as f64) - (sum_in as f64) * (sum_in as f64);
            if variance_part > 0.0 {
                (1.0 / variance_part.sqrt()) as f32
            } else {
                0.0
            }
        } else {
            // No squared integral image attached (e.g. demo cascade). Skip
            // variance normalisation — use raw feature response.
            1.0
        };
        if variance_norm_factor == 0.0 {
            return None;
        }

        let mut total: f32 = 0.0;
        cache.clear();
        // Hoist ii/ri dimensions to stack — they're used per-rect and would
        // otherwise be re-read inside the inner loop.
        let ii_w = ii.width();
        let ii_h = ii.height();
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
            total += stage_sum;
        }
        Some(total)
    }

    /// Save the cascade to a compact binary file.
    /// Format: magic "RFCF" u32, version=2, then feature + stage records.
    /// Version 2 uses f32 weights and supports arbitrary rectangle layouts.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"RFCF")?;
        f.write_all(&2u32.to_le_bytes())?; // version 2
        f.write_all(&(self.window_w as u32).to_le_bytes())?;
        f.write_all(&(self.window_h as u32).to_le_bytes())?;
        f.write_all(&(self.features.len() as u32).to_le_bytes())?;
        for feat in &self.features {
            f.write_all(&[feat.kind as u8, feat.width, feat.height])?;
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

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
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
        let mut features = Vec::with_capacity(nfeat);
        for _ in 0..nfeat {
            let mut head = [0u8; 3];
            f.read_exact(&mut head)?;
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
        let _ = version;
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
}
