//! Runtime-free face-crop quality gate (IQA for anti-spoofing).
//!
//! Silent RGB liveness such as MiniFASNet leans on high-frequency detail:
//! a heavily blurred, tiny, over/under-exposed or clipped crop has already
//! lost the cues that separate a live face from a print or a screen. The
//! published model still returns three confident logits for such a crop —
//! from the wrong distribution — so gating on raw image quality *before*
//! trusting the classifier is the cheapest, most reliable hardening step.
//!
//! Like [`crate::liveness`], this module holds no inference dependency and
//! ships in the default zero-dependency build. It measures:
//!
//! - **sharpness** — variance of the 4-neighbour Laplacian, the same focus
//!   statistic OpenCV's `cv2.Laplacian(..., cv2.CV_64F).var()` returns.
//! - **brightness** — mean luma, so dark/washed-out frames can be rejected.
//! - **clipping** — fraction of pixels pinned to `0`/`255`, which spikes on
//!   clipped phone replays and contrast-crushed prints.
//! - **resolution** — the crop edge length, so distant tiny faces are not
//!   scored from an `80×80` up-sample.
//!
//! The gate is deliberately **off by default** ([`QualityConfig::enabled`])
//! so existing callers keep their exact behaviour; opt in with [`QualityConfig::strict`]
//! or by setting fields explicitly.

use crate::image::GrayImage;

/// Tunable knobs for the quality gate.
#[derive(Clone, Debug)]
pub struct QualityConfig {
    /// Master switch. When `false`, [`assess`] always accepts the crop.
    pub enabled: bool,
    /// Minimum crop edge in pixels; `0` disables the size check.
    pub min_face_size: usize,
    /// Minimum Laplacian variance (focus) to accept.
    pub min_sharpness: f32,
    /// Minimum mean luma, `[0, 255]`.
    pub min_brightness: f32,
    /// Maximum mean luma, `[0, 255]`.
    pub max_brightness: f32,
    /// Maximum fraction `[0, 1]` of pixels clipped to `0`/`255`.
    pub max_clipped_ratio: f32,
}

impl Default for QualityConfig {
    /// Disabled: reproduce the historical "always trust the classifier" path.
    fn default() -> Self {
        Self {
            enabled: false,
            min_face_size: 0,
            min_sharpness: 0.0,
            min_brightness: 0.0,
            max_brightness: 255.0,
            max_clipped_ratio: 1.0,
        }
    }
}

impl QualityConfig {
    /// Conservative, security-oriented preset tuned for an RGB face crop.
    ///
    /// Thresholds here are deliberately moderate rather than aggressive:
    /// they reject clearly unusable frames without materially raising the
    /// false-reject rate on ordinary webcams. Calibrate against your own
    /// devices before tightening further.
    pub fn strict() -> Self {
        Self {
            enabled: true,
            min_face_size: 48,
            min_sharpness: 15.0,
            min_brightness: 25.0,
            max_brightness: 232.0,
            max_clipped_ratio: 0.35,
        }
    }
}

/// Why a crop failed the gate; reported to aid logging and threshold tuning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityIssue {
    /// Crop edge is shorter than [`QualityConfig::min_face_size`].
    TooSmall,
    /// Laplacian variance below [`QualityConfig::min_sharpness`].
    Blurry,
    /// Mean luma outside `[min_brightness, max_brightness]`.
    BadExposure,
    /// Too many pixels pinned to `0`/`255`.
    Clipped,
}

impl QualityIssue {
    /// Stable machine-readable identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            QualityIssue::TooSmall => "too_small",
            QualityIssue::Blurry => "blurry",
            QualityIssue::BadExposure => "bad_exposure",
            QualityIssue::Clipped => "clipped",
        }
    }
}

/// Measured quality statistics plus the gate verdict.
#[derive(Clone, Debug)]
pub struct QualityReport {
    /// Crop width in pixels.
    pub width: usize,
    /// Crop height in pixels.
    pub height: usize,
    /// Variance of the 4-neighbour Laplacian (focus measure).
    pub sharpness: f32,
    /// Mean luma in `[0, 255]`.
    pub mean_brightness: f32,
    /// Fraction `[0, 1]` of pixels clipped to `0`/`255`.
    pub clipped_ratio: f32,
    /// `true` when every enabled check passes (or the gate is disabled).
    pub acceptable: bool,
}

impl QualityReport {
    /// Evaluate the measured statistics against `cfg`, returning the issues
    /// found (empty when the crop is acceptable).
    pub fn issues(&self, cfg: &QualityConfig) -> Vec<QualityIssue> {
        let mut issues = Vec::new();
        if !cfg.enabled {
            return issues;
        }
        if cfg.min_face_size > 0
            && (self.width < cfg.min_face_size || self.height < cfg.min_face_size)
        {
            issues.push(QualityIssue::TooSmall);
        }
        if self.sharpness < cfg.min_sharpness {
            issues.push(QualityIssue::Blurry);
        }
        if self.mean_brightness < cfg.min_brightness || self.mean_brightness > cfg.max_brightness {
            issues.push(QualityIssue::BadExposure);
        }
        if self.clipped_ratio > cfg.max_clipped_ratio {
            issues.push(QualityIssue::Clipped);
        }
        issues
    }
}

/// Measure the quality statistics of a grayscale face crop.
///
/// The returned [`QualityReport::acceptable`] is the verdict under `cfg`;
/// a disabled config always yields `acceptable == true`. A `0`-area or
/// single-row crop is reported as unusable (zero sharpness) rather than
/// accepted, since it cannot carry face detail.
pub fn assess(gray: &GrayImage, cfg: &QualityConfig) -> QualityReport {
    let width = gray.width();
    let height = gray.height();
    let px = gray.as_slice();

    // Brightness and clipping are computed over every pixel.
    let mut luma_sum: u64 = 0;
    let mut clipped: u64 = 0;
    for &v in px {
        luma_sum += v as u64;
        if v == 0 || v == 255 {
            clipped += 1;
        }
    }
    let total = px.len().max(1) as f32;
    let mean_brightness = luma_sum as f32 / total;
    let clipped_ratio = clipped as f32 / total;

    // Laplacian variance needs a 1-pixel interior border.
    let interior = width.saturating_sub(2) * height.saturating_sub(2);
    let mut sharpness = 0.0f32;
    if interior > 0 {
        let mut sum: f32 = 0.0;
        let mut sum_sq: f32 = 0.0;
        for y in 1..height - 1 {
            let row = y * width;
            for x in 1..width - 1 {
                let center = px[row + x] as i32 * 4;
                let lap = center
                    - px[row + x - 1] as i32
                    - px[row + x + 1] as i32
                    - px[row - width + x] as i32
                    - px[row + width + x] as i32;
                let l = lap as f32;
                sum += l;
                sum_sq += l * l;
            }
        }
        let n = interior as f32;
        let mean = sum / n;
        sharpness = (sum_sq / n - mean * mean).max(0.0);
    }

    let mut report = QualityReport {
        width,
        height,
        sharpness,
        mean_brightness,
        clipped_ratio,
        acceptable: true,
    };
    report.acceptable = cfg.enabled && report.issues(cfg).is_empty() || !cfg.enabled;
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: usize, h: usize) -> GrayImage {
        // Smooth diagonal ramp: well-exposed, in range, but perfectly flat
        // gradients => Laplacian ~ 0 (blurry by the focus measure).
        let mut g = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                g.as_mut_slice()[y * w + x] = (((x + y) * 255 / (w + h - 2)) as u32).min(255) as u8;
            }
        }
        g
    }

    fn checker(w: usize, h: usize) -> GrayImage {
        // Alternating 0/255: maximum focus and maximum clipping.
        let mut g = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                g.as_mut_slice()[y * w + x] = if (x + y) % 2 == 0 { 255 } else { 0 };
            }
        }
        g
    }

    #[test]
    fn disabled_gate_accepts_anything() {
        let r = assess(&gradient(60, 60), &QualityConfig::default());
        assert!(r.acceptable);
    }

    #[test]
    fn sharpness_detects_flat_and_textured_crops() {
        let flat_img = GrayImage::from_vec(vec![120u8; 60 * 60], 60, 60);
        let flat = assess(&flat_img, &QualityConfig::strict());
        assert!(flat.sharpness < 1.0, "smooth ramp has ~zero focus");
        let textured = assess(&checker(60, 60), &QualityConfig::strict());
        assert!(textured.sharpness > 100.0, "checkerboard is very sharp");
    }

    #[test]
    fn rejects_blurry_but_exposed_crop() {
        let cfg = QualityConfig::strict();
        let r = assess(&gradient(60, 60), &cfg);
        let issues = r.issues(&cfg);
        assert!(issues.contains(&QualityIssue::Blurry));
        assert!(!r.acceptable);
    }

    #[test]
    fn rejects_tiny_crop() {
        let cfg = QualityConfig::strict();
        let r = assess(&checker(20, 20), &cfg);
        assert!(r.issues(&cfg).contains(&QualityIssue::TooSmall));
        assert!(!r.acceptable);
    }

    #[test]
    fn detects_clipping() {
        let cfg = QualityConfig {
            enabled: true,
            min_face_size: 0,
            min_sharpness: 0.0,
            min_brightness: 0.0,
            max_brightness: 255.0,
            max_clipped_ratio: 0.5,
        };
        // Checkerboard is 100% pixels at the extremes => ratio 1.0.
        let r = assess(&checker(40, 40), &cfg);
        assert!((r.clipped_ratio - 1.0).abs() < 1e-6);
        assert!(r.issues(&cfg).contains(&QualityIssue::Clipped));
    }

    #[test]
    fn detects_dark_and_bright_exposure() {
        let dark = GrayImage::from_vec(vec![5u8; 40 * 40], 40, 40);
        let cfg = QualityConfig {
            enabled: true,
            min_face_size: 0,
            min_sharpness: 0.0,
            min_brightness: 25.0,
            max_brightness: 232.0,
            max_clipped_ratio: 1.0,
        };
        assert!(assess(&dark, &cfg)
            .issues(&cfg)
            .contains(&QualityIssue::BadExposure));
        let bright = GrayImage::from_vec(vec![250u8; 40 * 40], 40, 40);
        assert!(assess(&bright, &cfg)
            .issues(&cfg)
            .contains(&QualityIssue::BadExposure));
    }

    #[test]
    fn accepts_midtone_textured_crop() {
        // Mid-gray with a sparse textured pattern: in-range, modest clipping,
        // non-zero focus, large enough.
        let mut g = GrayImage::from_vec(vec![128u8; 60 * 60], 60, 60);
        for y in (1..59).step_by(3) {
            for x in (1..59).step_by(3) {
                g.as_mut_slice()[y * 60 + x] = 160;
            }
        }
        let r = assess(&g, &QualityConfig::strict());
        assert!(
            r.acceptable,
            "unexpected issues: {:?}",
            r.issues(&QualityConfig::strict())
        );
    }

    #[test]
    fn degenerate_crop_is_unusable_when_enabled() {
        let g = GrayImage::new(2, 2);
        let r = assess(&g, &QualityConfig::strict());
        assert_eq!(r.sharpness, 0.0);
        assert!(!r.acceptable);
    }
}
