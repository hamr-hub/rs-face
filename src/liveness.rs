//! Silent face liveness — pure pre-processing and decision logic.
//!
//! This module is the runtime-free numerical core of the MiniVision
//! Silent-Face-Anti-Spoofing integration (Apache-2.0). Like `crate::scrfd`
//! and `crate::arcface`, it holds no reference to any inference backend:
//! the crop-expansion, NCHW build, softmax and the two-model fusion decision
//! live here as plain functions over slices, so the parts most likely to hide
//! a silent security bug are unit-testable without a model file. The ONNX
//! wiring lives in `crate::liveness_detector`.
//!
//! # What the models actually expect
//!
//! The published MiniFASNet ONNX exports take a raw **`[0, 255]`** tensor —
//! no mean subtraction, no `/127.5` — in **BGR** channel order (they were
//! trained on `cv2.imread` output), reshaped to `1×3×80×80`. Getting either
//! detail wrong does not error: the network still emits three confident
//! logits, just from the wrong distribution, weakening spoof rejection.
//!
//! # Output classes
//!
//! The three logits map to `[printed photo, real face, screen replay]`; the
//! real class is index [`REAL_CLASS`]. Two models look at different crop
//! scales ([`CROP_SCALE_V2`], [`CROP_SCALE_V1SE`]); their softmax outputs
//! are averaged before argmax, the upstream-recommended ensemble.

use crate::face::Detection;

/// Model input edge length; crops are resized to `INPUT × INPUT`.
pub const INPUT: usize = 80;
/// Number of output classes.
pub const NUM_CLASSES: usize = 3;
/// Class index of a real face: `[paper, real, screen]`.
pub const REAL_CLASS: usize = 1;
/// Crop-expansion scale for MiniFASNetV2.
pub const CROP_SCALE_V2: f32 = 2.7;
/// Crop-expansion scale for MiniFASNetV1SE.
pub const CROP_SCALE_V1SE: f32 = 4.0;

/// The canonical two-model crop scales, in model order.
pub const CROP_SCALES: [f32; 2] = [CROP_SCALE_V2, CROP_SCALE_V1SE];

/// Tunable knobs for the liveness decision.
#[derive(Clone, Debug)]
pub struct LivenessConfig {
    /// Minimum averaged real-class probability to accept a face as live.
    ///
    /// `0.0` reproduces the upstream pure-argmax rule (real just has to beat
    /// the two spoof classes); raise it toward `0.5`+ for a stricter gate at
    /// the cost of a higher false-reject rate.
    pub min_real_score: f32,
}

impl Default for LivenessConfig {
    fn default() -> Self {
        Self {
            min_real_score: 0.0,
        }
    }
}

/// Compute the expanded, boundary-clamped crop rectangle for a detection.
///
/// Mirrors the reference `_get_new_box`: the face box is grown by `scale`
/// about its centre, the scale is first limited so it cannot exceed the image
/// bounds, and each side that would leave the frame pushes the opposite side
/// back. Returns `(x, y, width, height)` in **exclusive** pixel coordinates
/// suitable for [`crate::image::RgbImage::crop`].
pub fn expanded_crop(
    src_w: usize,
    src_h: usize,
    det: &Detection,
    scale: f32,
) -> (usize, usize, usize, usize) {
    if src_w == 0 || src_h == 0 || det.w == 0 || det.h == 0 {
        return (0, 0, 0, 0);
    }
    let bw = det.w as f32;
    let bh = det.h as f32;
    let limited = scale
        .min((src_w as f32 - 1.0) / bw)
        .min((src_h as f32 - 1.0) / bh);
    let new_w = bw * limited;
    let new_h = bh * limited;
    let cx = det.x as f32 + bw / 2.0;
    let cy = det.y as f32 + bh / 2.0;

    let mut l = cx - new_w / 2.0;
    let mut t = cy - new_h / 2.0;
    let mut r = cx + new_w / 2.0;
    let mut b = cy + new_h / 2.0;

    // Boundary handling: clamp one side and shift the opposite side by the
    // overflow so the crop keeps its size while staying in the frame.
    if l < 0.0 {
        r -= l;
        l = 0.0;
    }
    if t < 0.0 {
        b -= t;
        t = 0.0;
    }
    let max_x = src_w as f32 - 1.0;
    let max_y = src_h as f32 - 1.0;
    if r > max_x {
        l -= r - max_x;
        r = max_x;
    }
    if b > max_y {
        t -= b - max_y;
        b = max_y;
    }
    l = l.max(0.0);
    t = t.max(0.0);
    r = r.min(max_x);
    b = b.min(max_y);

    // Convert the inclusive reference box to an exclusive crop rectangle.
    let x0 = l.floor() as usize;
    let y0 = t.floor() as usize;
    let x1 = (r.ceil() as usize + 1).min(src_w);
    let y1 = (b.ceil() as usize + 1).min(src_h);
    (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
}

/// Build the model input tensor from an already-resized 80×80 **RGB** crop.
///
/// The crop is de-interleaved into NCHW planes in **BGR** order and kept in
/// the raw `[0, 255]` range the exported models expect. Returns `None` if the
/// crop is not exactly [`INPUT`] square — silently resizing here would mask a
/// bug in the caller.
pub fn preprocess_bgr(crop: &crate::image::RgbImage) -> Option<Vec<f32>> {
    if crop.width() != INPUT || crop.height() != INPUT {
        return None;
    }
    let plane = INPUT * INPUT;
    let mut out = vec![0.0f32; 3 * plane];
    let src = crop.as_slice();
    for i in 0..plane {
        out[0 * plane + i] = src[i * 3 + 2] as f32; // B
        out[1 * plane + i] = src[i * 3 + 1] as f32; // G
        out[2 * plane + i] = src[i * 3 + 0] as f32; // R
    }
    Some(out)
}

/// Numerically stable softmax over a fixed-width logit slice.
///
/// Returns `None` if the slice is not [`NUM_CLASSES`] wide or contains a
/// non-finite value, so a degenerate model output can never become a spoof
/// verdict rather than failing.
pub fn softmax(logits: &[f32]) -> Option<[f32; NUM_CLASSES]> {
    if logits.len() != NUM_CLASSES || logits.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let lmax = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = [0.0f32; NUM_CLASSES];
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        let e = (v - lmax).exp();
        probs[i] = e;
        sum += e;
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }
    Some(probs)
}

/// The outcome of a liveness check.
#[derive(Clone, Debug)]
pub struct LivenessOutcome {
    /// `true` when the averaged decision accepts a live face.
    pub is_real: bool,
    /// Averaged probability of the real class.
    pub real_score: f32,
    /// Averaged class probabilities `[paper, real, screen]`.
    pub probs: [f32; NUM_CLASSES],
    /// Winning class index.
    pub class: usize,
    /// Measured native-crop quality statistics when the detector computed
    /// them (`sharpness`, `high_freq_ratio`, exposure, clipping). Surfaced so
    /// callers can collect the signals alongside each verdict for threshold
    /// calibration; `None` for outcomes built without a quality measurement.
    pub quality: Option<crate::quality::QualityReport>,
}

impl LivenessOutcome {
    /// Human-readable attack / real label.
    pub fn label(&self) -> &'static str {
        match self.class {
            0 => "printed photo",
            1 => "real face",
            2 => "screen replay",
            // Sentinel produced by the quality gate (never by `decide`):
            // the crop was too poor to classify and was rejected fail-closed.
            LOW_QUALITY_CLASS => "low quality",
            _ => "unknown",
        }
    }
}

/// Pseudo-class marking a crop rejected by the quality gate rather than by
/// the classifier. Kept outside `0..NUM_CLASSES` so it can never be confused
/// with a real model output.
pub const LOW_QUALITY_CLASS: usize = NUM_CLASSES;

/// Fuse per-model softmax outputs by averaging and render the decision.
///
/// `rows` is one probability row per model (two for the canonical setup); an
/// empty set or a malformed row yields `None`. The face is accepted only when
/// the averaged argmax is the real class **and** the real probability clears
/// [`LivenessConfig::min_real_score`].
pub fn decide(rows: &[[f32; NUM_CLASSES]], config: &LivenessConfig) -> Option<LivenessOutcome> {
    if rows.is_empty() {
        return None;
    }
    let mut avg = [0.0f32; NUM_CLASSES];
    for row in rows {
        if row.iter().any(|v| !v.is_finite()) {
            return None;
        }
        for k in 0..NUM_CLASSES {
            avg[k] += row[k];
        }
    }
    let n = rows.len() as f32;
    for v in avg.iter_mut() {
        *v /= n;
    }
    let mut class = 0usize;
    for k in 1..NUM_CLASSES {
        if avg[k] > avg[class] {
            class = k;
        }
    }
    let real_score = avg[REAL_CLASS];
    let is_real = class == REAL_CLASS && real_score >= config.min_real_score;
    Some(LivenessOutcome {
        is_real,
        real_score,
        probs: avg,
        class,
        quality: None,
    })
}

/// Multi-frame temporal voting configuration.
///
/// A single frame can be fooled by a momentarily convincing replay; asking
/// for several *consecutive* real verdicts is the cheapest temporal defence
/// and is how the upstream project recommends driving a live stream. The
/// gate is **fail-closed**: until enough frames have been observed, the face
/// is not confirmed, so a clipped/too-short clip can never slip through with
/// only one good frame.
#[derive(Clone, Debug)]
pub struct TemporalConfig {
    /// Consecutive real frames required before a face is confirmed.
    ///
    /// `1` reproduces the plain per-frame behaviour (no temporal gate).
    pub required_frames: usize,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self { required_frames: 1 }
    }
}

/// Sliding window of consecutive per-frame liveness verdicts for one track.
///
/// Feed it frame by frame with [`TemporalVote::observe`]; a single non-real
/// frame resets the streak. Read [`TemporalVote::confirmed`] for the
/// fail-closed decision and [`TemporalVote::mean_real_score`] for a steadier
/// confidence than any one frame provides.
#[derive(Clone, Debug, Default)]
pub struct TemporalVote {
    required_frames: usize,
    streak: Vec<(bool, f32)>,
}

impl TemporalVote {
    /// Construct a vote gate. `required == 0` is treated as `1` so the gate
    /// always has a meaningful threshold.
    pub fn new(config: &TemporalConfig) -> Self {
        Self {
            required_frames: config.required_frames.max(1),
            streak: Vec::new(),
        }
    }

    /// Record one frame's verdict.
    ///
    /// A non-real frame breaks the consecutive-real streak (the window is
    /// cleared); a real frame appends its real-class probability.
    pub fn observe(&mut self, is_real: bool, real_score: f32) {
        if !is_real {
            self.streak.clear();
            return;
        }
        if self.streak.len() >= self.required_frames {
            self.streak.remove(0);
        }
        self.streak.push((true, real_score));
    }

    /// Number of consecutive real frames currently held.
    pub fn consecutive_real(&self) -> usize {
        self.streak.len()
    }

    /// Fail-closed confirmation: `true` only once at least
    /// [`TemporalConfig::required_frames`] consecutive frames were real.
    pub fn confirmed(&self) -> bool {
        self.streak.len() >= self.required_frames
    }

    /// Mean real-class probability over the current streak (`0.0` when empty).
    pub fn mean_real_score(&self) -> f32 {
        if self.streak.is_empty() {
            return 0.0;
        }
        self.streak.iter().map(|(_, s)| s).sum::<f32>() / self.streak.len() as f32
    }

    /// Drop all history (e.g. when the track leaves the frame).
    pub fn reset(&mut self) {
        self.streak.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::RgbImage;

    fn det(x: usize, y: usize, w: usize, h: usize) -> Detection {
        Detection {
            x,
            y,
            w,
            h,
            score: 1.0,
        }
    }

    #[test]
    fn expanded_crop_grows_and_stays_in_frame() {
        let d = det(40, 40, 20, 20);
        let (x, y, w, h) = expanded_crop(200, 200, &d, CROP_SCALE_V2);
        assert!(w > 20 && h > 20, "crop must be larger than the face");
        assert_eq!(w, h, "uniform scale keeps a square crop");
        assert!(x + w <= 200 && y + h <= 200, "crop stays in frame");
    }

    #[test]
    fn expanded_crop_clamps_at_image_bounds() {
        let d = det(0, 0, 30, 30);
        let (x, y, w, h) = expanded_crop(100, 100, &d, CROP_SCALE_V1SE);
        assert_eq!(x, 0, "near-edge face keeps the crop anchored at 0");
        assert_eq!(y, 0);
        assert!(x + w <= 100 && y + h <= 100, "never overflows");
    }

    #[test]
    fn preprocess_emits_bgr_nchw_raw_range() {
        let mut img = RgbImage::new(INPUT, INPUT);
        // Pixel 0 is pure red in RGB.
        img.as_mut_slice()[0] = 255; // R
        img.as_mut_slice()[1] = 0; // G
        img.as_mut_slice()[2] = 0; // B
        let t = preprocess_bgr(&img).expect("80x80 crop preprocesses");
        assert_eq!(t.len(), 3 * INPUT * INPUT);
        let plane = INPUT * INPUT;
        assert_eq!(t[2 * plane], 255.0, "R source lands in plane 2");
        assert_eq!(t[0], 0.0, "B plane first");
        assert_eq!(t[plane], 0.0, "G plane middle");
    }

    #[test]
    fn preprocess_rejects_wrong_size() {
        assert!(preprocess_bgr(&RgbImage::new(40, 40)).is_none());
    }

    #[test]
    fn softmax_normalises_and_picks_peak() {
        let p = softmax(&[1.0, 5.0, 1.0]).expect("softmax");
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(p[REAL_CLASS] > p[0] && p[REAL_CLASS] > p[2]);
    }

    #[test]
    fn softmax_rejects_bad_width_and_nonfinite() {
        assert!(softmax(&[1.0, 2.0]).is_none());
        assert!(softmax(&[1.0, f32::NAN, 0.0]).is_none());
    }

    #[test]
    fn decide_averages_and_accepts_real_majority() {
        let a = softmax(&[0.0, 4.0, 0.0]).unwrap();
        let b = softmax(&[0.0, 3.0, 1.0]).unwrap();
        let out = decide(&[a, b], &LivenessConfig::default()).expect("decision");
        assert!(out.is_real);
        assert_eq!(out.class, REAL_CLASS);
        assert!((out.real_score - (a[1] + b[1]) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn decide_flags_print_and_screen_attacks() {
        let paper = softmax(&[5.0, 0.0, 0.0]).unwrap();
        assert!(
            !decide(&[paper], &LivenessConfig::default())
                .unwrap()
                .is_real
        );
        let screen = softmax(&[0.0, 0.0, 5.0]).unwrap();
        assert!(
            !decide(&[screen], &LivenessConfig::default())
                .unwrap()
                .is_real
        );
    }

    #[test]
    fn decide_splits_disagreement_by_averaged_margin() {
        // Head A leans paper, head B leans real. Averaged: paper 0.50,
        // real 0.40, screen 0.10 — the attack class must win even though
        // one head judged the face real.
        let paper_lean = [0.70f32, 0.20, 0.10];
        let real_lean = [0.30f32, 0.60, 0.10];
        let out = decide(&[paper_lean, real_lean], &LivenessConfig::default())
            .expect("decision on disagreement");
        assert!((out.real_score - 0.40).abs() < 1e-6);
        assert!(!out.is_real, "higher averaged attack class wins");
        assert_eq!(out.class, 0);

        // A stronger real head tips the averaged real class on top:
        // paper 0.40, real 0.50, screen 0.10.
        let real_strong = [0.10f32, 0.80, 0.10];
        let flipped = decide(&[paper_lean, real_strong], &LivenessConfig::default())
            .expect("decision on flipped disagreement");
        assert!(flipped.is_real, "averaged real class now wins");
        assert_eq!(flipped.class, REAL_CLASS);
    }

    #[test]
    fn decide_honours_min_real_score() {
        // Real is argmax but only weakly: [0.34, 0.40, 0.26].
        let weak = [0.34f32, 0.40, 0.26];
        let loose = decide(
            &[weak],
            &LivenessConfig {
                min_real_score: 0.0,
            },
        )
        .unwrap();
        assert!(loose.is_real, "argmax rule accepts weak real");
        let strict = decide(
            &[weak],
            &LivenessConfig {
                min_real_score: 0.5,
            },
        );
        assert!(!strict.unwrap().is_real, "strict gate rejects weak real");
    }

    #[test]
    fn decide_rejects_empty_and_bad_rows() {
        assert!(decide(&[], &LivenessConfig::default()).is_none());
        assert!(decide(&[[0.0, f32::NAN, 0.0]], &LivenessConfig::default()).is_none());
    }

    #[test]
    fn temporal_is_per_frame_with_single_requirement() {
        let mut v = TemporalVote::new(&TemporalConfig { required_frames: 1 });
        v.observe(true, 0.9);
        assert!(v.confirmed());
        assert_eq!(v.consecutive_real(), 1);
        assert!((v.mean_real_score() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn temporal_requires_consecutive_real_frames() {
        let mut v = TemporalVote::new(&TemporalConfig { required_frames: 3 });
        v.observe(true, 0.6);
        assert!(!v.confirmed(), "fail-closed until 3 frames");
        v.observe(true, 0.7);
        assert!(!v.confirmed());
        v.observe(true, 0.8);
        assert!(v.confirmed(), "3 consecutive real frames confirm");
        assert!((v.mean_real_score() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn temporal_resets_on_any_spoof_frame() {
        let mut v = TemporalVote::new(&TemporalConfig { required_frames: 3 });
        v.observe(true, 0.8);
        v.observe(true, 0.8);
        v.observe(false, 0.1);
        assert_eq!(v.consecutive_real(), 0);
        assert!(!v.confirmed());
        v.observe(true, 0.9);
        assert_eq!(v.consecutive_real(), 1, "count restarts after reset");
    }

    #[test]
    fn temporal_keeps_only_the_latest_window() {
        let mut v = TemporalVote::new(&TemporalConfig { required_frames: 2 });
        for _ in 0..5 {
            v.observe(true, 0.9);
        }
        assert_eq!(v.consecutive_real(), 2, "window never exceeds required");
        assert!(v.confirmed());
    }

    #[test]
    fn temporal_zero_requirement_treated_as_one() {
        let mut v = TemporalVote::new(&TemporalConfig { required_frames: 0 });
        v.observe(true, 0.5);
        assert!(v.confirmed());
    }
}
