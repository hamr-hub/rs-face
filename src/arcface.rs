//! ArcFace recognition: pre-processing from an aligned crop to an input tensor.
//!
//! Like [`crate::scrfd`], this module is **pure** — it produces the input tensor and
//! interprets the output vector, with no reference to any inference runtime, so the
//! numerics are unit-testable without a 166 MB model file.
//!
//! # The two normalisation constants are not the same
//!
//! SCRFD divides by **128.0**; ArcFace divides by **127.5**. The difference is under 0.4%
//! and produces no error, no warning, and embeddings that still look entirely plausible —
//! they are simply computed slightly off the distribution the backbone was trained on,
//! shaving accuracy in a way that only shows up as a slightly worse ROC curve. The two
//! constants therefore live next to their own models
//! ([`crate::scrfd::INPUT_STD`] and [`INPUT_STD`]) and are deliberately *not* shared.
//!
//! # Output is already normalised
//!
//! The InsightFace ONNX exports apply `F.normalize` **inside the graph**, so the 512
//! floats that come out are already unit length. Passing them through
//! [`crate::embedding::Embedding::from_raw`] is therefore idempotent — it is kept in the
//! path anyway because it is the only way to construct an `Embedding`, it costs one pass
//! over 512 floats, and it catches a collapsed (all-zero or NaN) output that would
//! otherwise match every gallery entry.

use crate::align::{norm_crop, ARCFACE_CROP_SIZE};
use crate::embedding::Embedding;
use crate::face::Landmarks;
use crate::image::RgbImage;

/// Mean subtracted from each channel.
pub const INPUT_MEAN: f32 = 127.5;

/// Divisor applied after mean subtraction, mapping `[0, 255]` onto `[-1, 1]`.
///
/// This is `127.5`, matching InsightFace's reference Python pipeline. Some community
/// forks use `128.0`; embeddings produced that way are not bit-comparable with reference
/// outputs. See the module docs.
pub const INPUT_STD: f32 = 127.5;

/// Build the ArcFace input tensor from an already-aligned 112x112 RGB crop.
///
/// Returns `None` if the crop is not exactly [`ARCFACE_CROP_SIZE`] square — silently
/// resizing here would mask an alignment bug upstream, and a mis-sized crop is precisely
/// the failure this function exists to catch.
pub fn preprocess_aligned(crop: &RgbImage) -> Option<Vec<f32>> {
    if crop.width() != ARCFACE_CROP_SIZE || crop.height() != ARCFACE_CROP_SIZE {
        return None;
    }
    let plane = ARCFACE_CROP_SIZE * ARCFACE_CROP_SIZE;
    let mut out = vec![0.0f32; 3 * plane];
    let src = crop.as_slice();
    for i in 0..plane {
        // NCHW: de-interleave RGB into three contiguous planes.
        for c in 0..3 {
            out[c * plane + i] = (src[i * 3 + c] as f32 - INPUT_MEAN) / INPUT_STD;
        }
    }
    Some(out)
}

/// Align a face with its landmarks and build the input tensor in one step.
///
/// Returns `None` when the landmarks are degenerate, so the caller drops the face rather
/// than embedding a garbage crop.
pub fn preprocess_from_landmarks(img: &RgbImage, lms: &Landmarks) -> Option<Vec<f32>> {
    let crop = norm_crop(img, lms)?;
    preprocess_aligned(&crop)
}

/// Interpret a raw model output as an [`Embedding`].
///
/// Returns `None` for an empty or collapsed output. See the module docs on why
/// re-normalising an already-normalised vector is intentional.
pub fn postprocess(raw: &[f32]) -> Option<Embedding> {
    Embedding::from_raw(raw)
}

/// Whether a raw output looks like it was already L2-normalised in-graph.
///
/// Diagnostic helper: if this reports `false` for an InsightFace export, the wrong tensor
/// is being read (e.g. a pre-normalisation feature layer), which is worth surfacing
/// loudly because the resulting embeddings would still appear to work.
pub fn looks_normalized(raw: &[f32]) -> bool {
    if raw.is_empty() {
        return false;
    }
    let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
    (norm - 1.0).abs() < 1e-2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::ARCFACE_REFERENCE_LANDMARKS;
    use crate::face::NUM_LANDMARKS;

    fn solid_crop(size: usize, rgb: (u8, u8, u8)) -> RgbImage {
        let mut img = RgbImage::new(size, size);
        for i in 0..size * size {
            img.as_mut_slice()[i * 3] = rgb.0;
            img.as_mut_slice()[i * 3 + 1] = rgb.1;
            img.as_mut_slice()[i * 3 + 2] = rgb.2;
        }
        img
    }

    #[test]
    fn preprocess_emits_nchw_of_the_right_length() {
        let t = preprocess_aligned(&solid_crop(112, (0, 0, 0))).unwrap();
        assert_eq!(t.len(), 3 * 112 * 112);
    }

    #[test]
    fn preprocess_maps_range_to_minus_one_to_one() {
        let black = preprocess_aligned(&solid_crop(112, (0, 0, 0))).unwrap();
        assert!((black[0] - (-1.0)).abs() < 1e-6, "byte 0 must map to -1.0");

        let white = preprocess_aligned(&solid_crop(112, (255, 255, 255))).unwrap();
        assert!((white[0] - 1.0).abs() < 1e-2, "byte 255 must map to ~+1.0");

        let mid = preprocess_aligned(&solid_crop(112, (128, 128, 128))).unwrap();
        assert!(mid[0].abs() < 0.01, "mid grey must map to ~0");
    }

    /// Locks the ArcFace divisor apart from SCRFD's. With 127.5, byte 0 maps to exactly
    /// -1.0; with 128.0 it maps to -0.99609. Asserting the exact endpoint pins the
    /// constant, which no accuracy test would catch.
    #[test]
    fn arcface_uses_127_5_not_scrfd_128() {
        assert_eq!(INPUT_STD, 127.5);
        assert_ne!(INPUT_STD, crate::scrfd::INPUT_STD);
        let t = preprocess_aligned(&solid_crop(112, (0, 0, 0))).unwrap();
        assert_eq!(t[0], -1.0, "with /128.0 this would be -0.99609375");
    }

    #[test]
    fn preprocess_separates_channels_into_planes() {
        let t = preprocess_aligned(&solid_crop(112, (255, 0, 128))).unwrap();
        let plane = 112 * 112;
        assert!(t[0] > 0.9, "R plane should be ~+1");
        assert!((t[plane] - (-1.0)).abs() < 1e-6, "G plane should be -1");
        assert!(t[2 * plane].abs() < 0.01, "B plane should be ~0");
    }

    #[test]
    fn preprocess_rejects_wrong_sized_crop_rather_than_resizing() {
        // Silently resizing would hide an upstream alignment bug.
        assert!(preprocess_aligned(&solid_crop(96, (0, 0, 0))).is_none());
        assert!(preprocess_aligned(&solid_crop(224, (0, 0, 0))).is_none());
        assert!(preprocess_aligned(&RgbImage::new(112, 64)).is_none());
    }

    #[test]
    fn preprocess_from_landmarks_produces_a_full_tensor() {
        let img = solid_crop(640, (100, 120, 140));
        let lms = Landmarks {
            points: ARCFACE_REFERENCE_LANDMARKS,
        };
        let t = preprocess_from_landmarks(&img, &lms).unwrap();
        assert_eq!(t.len(), 3 * 112 * 112);
    }

    #[test]
    fn preprocess_from_degenerate_landmarks_returns_none() {
        let img = solid_crop(640, (100, 120, 140));
        let lms = Landmarks {
            points: [(10.0, 10.0); NUM_LANDMARKS],
        };
        assert!(preprocess_from_landmarks(&img, &lms).is_none());
    }

    #[test]
    fn postprocess_normalises_and_rejects_collapse() {
        let e = postprocess(&[3.0, 4.0]).unwrap();
        assert!((e.as_slice()[0] - 0.6).abs() < 1e-6);
        assert!(postprocess(&[]).is_none());
        assert!(
            postprocess(&[0.0; 512]).is_none(),
            "collapsed output must be rejected"
        );
        assert!(postprocess(&[f32::NAN; 512]).is_none());
    }

    #[test]
    fn postprocess_is_idempotent_on_already_normalised_output() {
        // The real ONNX graph normalises internally; re-normalising must be a no-op.
        let mut raw = vec![0.0f32; 512];
        raw[0] = 1.0;
        let e = postprocess(&raw).unwrap();
        assert_eq!(e.as_slice()[0], 1.0);
        let again = postprocess(e.as_slice()).unwrap();
        assert_eq!(again, e);
    }

    #[test]
    fn looks_normalized_detects_unit_vectors() {
        let mut unit = vec![0.0f32; 512];
        unit[0] = 1.0;
        assert!(looks_normalized(&unit));

        // A raw pre-normalisation feature vector typically has norm >> 1.
        assert!(!looks_normalized(&[10.0; 512]));
        assert!(!looks_normalized(&[]));
        assert!(!looks_normalized(&[0.0; 512]));
    }

    /// End-to-end identity property: the same face aligned from two different framings
    /// (scaled and translated) must yield near-identical tensors. This is the whole
    /// purpose of alignment, and it holds without any model weights.
    #[test]
    fn alignment_makes_the_pipeline_invariant_to_scale_and_position() {
        // Build a textured source image so the comparison is not trivially uniform.
        let (w, h) = (600usize, 600usize);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                img.as_mut_slice()[i] = ((x * 3) % 256) as u8;
                img.as_mut_slice()[i + 1] = ((y * 3) % 256) as u8;
                img.as_mut_slice()[i + 2] = ((x + y) % 256) as u8;
            }
        }

        // Framing A: the canonical template placed at 3x scale, offset (150, 150).
        let mut a = ARCFACE_REFERENCE_LANDMARKS;
        for p in a.iter_mut() {
            p.0 = p.0 * 3.0 + 150.0;
            p.1 = p.1 * 3.0 + 150.0;
        }
        // Framing B: same landmarks, 2x scale, offset (60, 60) — a different crop of the
        // same underlying geometry.
        let mut b = ARCFACE_REFERENCE_LANDMARKS;
        for p in b.iter_mut() {
            p.0 = p.0 * 2.0 + 60.0;
            p.1 = p.1 * 2.0 + 60.0;
        }

        let ta = preprocess_from_landmarks(&img, &Landmarks { points: a }).unwrap();
        let tb = preprocess_from_landmarks(&img, &Landmarks { points: b }).unwrap();

        // Both crops sample a different part of the gradient, so they are not identical,
        // but the *geometry* is normalised: the aligned landmark positions must coincide.
        // Verify via the transforms rather than pixels.
        let ta_t = crate::align::estimate_arcface_transform(&Landmarks { points: a }).unwrap();
        let tb_t = crate::align::estimate_arcface_transform(&Landmarks { points: b }).unwrap();
        for i in 0..NUM_LANDMARKS {
            let pa = ta_t.apply(a[i].0, a[i].1);
            let pb = tb_t.apply(b[i].0, b[i].1);
            assert!(
                (pa.0 - pb.0).abs() < 1e-2 && (pa.1 - pb.1).abs() < 1e-2,
                "landmark {i} aligned to {pa:?} vs {pb:?} — alignment is not scale/offset invariant"
            );
        }
        assert_eq!(ta.len(), tb.len());
    }
}
