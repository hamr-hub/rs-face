//! Pixel-accuracy regression tests for the hand-rolled JPEG decoder.
//!
//! # Why this file exists
//!
//! The decoder in `src/image/jpeg.rs` was, for a long time, producing complete
//! garbage: a 16x16 block mosaic where a photograph should have been, because
//! the IDCT was 4x too large, the de-zigzag was applied backwards, every
//! subsampled chroma block was written at twice its correct offset, and files
//! with restart intervals were truncated at the first RSTn marker.
//!
//! Every one of those shipped. They shipped because the pre-existing tests in
//! `tests/jpeg_decode.rs` only assert two things: that the output has the
//! right *dimensions*, and that it is "not a uniform colour". A 16x16 mosaic
//! passes both checks perfectly. A dimension assertion cannot distinguish a
//! correct decode from noise, so no amount of adding more of them would ever
//! have caught this.
//!
//! These tests assert the only property that actually matters — that the
//! decoded *pixels* match a reference decoder.
//!
//! # The oracle
//!
//! The `.raw` files alongside each `.jpg` are PIL/libjpeg decodes of that same
//! JPEG: row-major RGB, 8 bits per channel. libjpeg is the oracle because it
//! is what every other tool agrees with, so agreement with it is the strongest
//! statement we can make without pulling in a second independent decoder.
//!
//! Byte equality is deliberately *not* required. We compute the IDCT in f64
//! where libjpeg uses a fixed-point approximation, and our chroma upsampling
//! rounds once where libjpeg rounds between its two separable passes. Both are
//! worth about half a level. `MIN_PSNR_DB` is set at 40 dB, which is an RMS
//! error of ~2.5 levels — loose enough to absorb that rounding, tight enough
//! that nothing short of a real decoding error gets through.
//!
//! # Regenerating
//!
//! ```text
//! python3 tools/eval_m1/make_jpeg_corpus.py
//! cp data/eval_jpeg/<name>.{jpg,raw} tests/fixtures/jpeg_oracle/
//! ```
//!
//! The larger corpus (real photographs, every subsampling, more odd sizes) is
//! kept out of git and graded by `tools/eval_m1/jpeg_oracle.rs`, which prints a
//! per-case PSNR table. Run it after any change to the decoder.

use rsface::image::jpeg::decode_jpeg_rgb;

/// Minimum acceptable PSNR against the libjpeg reference. See module docs.
const MIN_PSNR_DB: f64 = 40.0;

/// (name, expected width, expected height) — sizes are the reference's.
const CASES: &[(&str, usize, usize)] = &[
    // 4:4:4 — no chroma resampling at all, so this isolates the entropy
    // decode + dequantise + IDCT path from the upsampler entirely.
    ("synth_grid__q95_444", 64, 48),
    ("synth_gradient__q95_444", 64, 48),
    // 4:2:2 — chroma subsampled horizontally only. Catches any code that
    // assumes the 4:2:0 2x-in-both-axes case.
    ("synth_grid__q95_422", 64, 48),
    // 4:2:0 — the common case.
    ("synth_grid__q95_420", 64, 48),
    // Restart intervals (DRI/RSTn). Every block resets its DC predictor and
    // the bit stream re-aligns on a byte boundary at each marker.
    ("synth_grid__q95_420_restart", 64, 48),
    // Single-component grayscale — no colour conversion or upsampling.
    ("synth_grid__q95_gray", 64, 48),
    // High-frequency content at heavy quantisation. The most sensitive probe
    // of the zig-zag order: a transposed permutation still "looks" plausible on
    // a smooth image, because position 0 is a fixed point of the permutation,
    // but wrecks a checkerboard.
    ("synth_checker__q30_420", 64, 48),
    // Not a multiple of the MCU size — exercises the padding and crop path.
    ("odd_17x13__q95_420", 17, 13),
];

fn psnr_and_max_err(got: &[u8], want: &[u8]) -> (f64, u32) {
    assert_eq!(got.len(), want.len(), "decoded length differs from reference");
    let n = got.len() as f64;
    let mut se = 0.0f64;
    let mut max_err = 0u32;
    for i in 0..got.len() {
        let d = got[i].abs_diff(want[i]) as f64;
        se += d * d;
        max_err = max_err.max(got[i].abs_diff(want[i]) as u32);
    }
    let mse = se / n;
    // mse == 0 means an exact match; report it as a large finite value so the
    // assertion below reads the same either way.
    let psnr = if mse <= f64::MIN_POSITIVE {
        200.0
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    };
    (psnr, max_err)
}

#[test]
fn decodes_to_match_the_libjpeg_reference() {
    for &(name, w, h) in CASES {
        let jpg = match std::fs::read(format!("tests/fixtures/jpeg_oracle/{name}.jpg")) {
            Ok(b) => b,
            // The fixtures are small binaries in-tree; a missing one is a
            // packaging bug, not a reason to silently skip the check.
            Err(e) => panic!("missing fixture {name}.jpg: {e}"),
        };
        let want = match std::fs::read(format!("tests/fixtures/jpeg_oracle/{name}.raw")) {
            Ok(b) => b,
            Err(e) => panic!("missing reference {name}.raw: {e}"),
        };

        let img = decode_jpeg_rgb(&jpg)
            .unwrap_or_else(|e| panic!("{name}: decode failed outright: {e}"));

        assert_eq!(
            (img.width(), img.height()),
            (w, h),
            "{name}: wrong output dimensions"
        );
        assert_eq!(
            want.len(),
            w * h * 3,
            "{name}: reference file is not w*h*3 bytes"
        );

        let (psnr, max_err) = psnr_and_max_err(img.as_slice(), &want);
        assert!(
            psnr >= MIN_PSNR_DB,
            "{name}: PSNR {psnr:.2} dB is below the {MIN_PSNR_DB} dB floor \
             (max per-channel error {max_err}). The decoded pixels do not match \
             the reference decoder — check the IDCT scale, the zig-zag \
             direction, chroma block placement and restart handling before \
             suspecting anything else.",
        );
    }
}

/// The decoded vertical intensity profile must be smooth.
///
/// The PSNR test above is a strong check but it fails without saying *which*
/// stage broke. This one is deliberately narrow and diagnostic. The source is
/// a smooth radial gradient, so consecutive rows must differ only slightly.
/// A transposed zig-zag, a misplaced block, or a mis-scaled DC shows up
/// immediately as a row-to-row jump — which names the bug rather than just
/// reporting a number.
#[test]
fn vertical_intensity_profile_is_smooth() {
    let name = "synth_gradient__q95_444";
    let jpg = std::fs::read(format!("tests/fixtures/jpeg_oracle/{name}.jpg")).expect("fixture");
    let img = decode_jpeg_rgb(&jpg).expect("decode");
    let (w, h) = (img.width(), img.height());
    let s = img.as_slice();
    let row_mean = |y: usize| -> f64 {
        (0..w).map(|x| s[(y * w + x) * 3] as f64).sum::<f64>() / w as f64
    };
    let profile: Vec<f64> = (0..h).map(row_mean).collect();
    for y in 1..h {
        let diff = (profile[y] - profile[y - 1]).abs();
        assert!(
            diff < 25.0,
            "row {y} mean ({:.1}) jumps {:.1} from row {} ({:.1}) — \
             a smooth gradient should not do that. A zig-zag, block-placement \
             or DC-scale error produces exactly this row-to-row jump. \
             profile={profile:?}",
            profile[y],
            diff,
            y - 1,
            profile[y - 1],
        );
    }
}
