# Zero-dependency face recognition: LBPH

This document describes the recogniser that ships in the crate's **default,
zero-third-party-dependency** build and reports its measured accuracy on real faces.

## 1. What "zero dependencies" means here

| Capability | Default build (`--no-default-features`) | Optional (`--features onnx`) |
|---|---|---|
| Detection | Viola–Jones Haar cascade (pure Rust) | SCRFD-10G (ONNX, 17 MB weights) |
| Recognition | **LBPH — no weights, no crates** | ArcFace R50 (ONNX, 174 MB weights) |

LBPH ([Ahonen, Hadid, Pietikäinen, ECCV 2004](https://link.springer.com/chapter/10.1007/978-3-540-24670-1_36))
needs no training-time download: the "model" is the set of per-identity histograms
collected at enrolment. The implementation lives in `src/lbph.rs` and uses only
`std` plus image primitives already in `src/image/`. The descriptor, the uniform-pattern
lookup table, and the chi-square metric are binary-compatible in spirit with OpenCV's
`face::LBPHFaceRecognizer` (radius 1, 8 neighbours, 59-bin uniform mapping, 8×8 cells,
chi-square distance), so published LBPH operating experience transfers.

## 2. Pipeline

```text
gray frame
  └─ face box (Haar in the zero-dep build; SCRFD when a model is present)
       └─ square box crop, +30 % margin, resize to 120×120
            └─ LBP codes (3×3, 8 neighbours, threshold vs. centre)
                 └─ map to 59 uniform bins
                      └─ 8×8 spatial cell histograms, L1-normalised per cell
                           └─ χ² distance Σ(a−b)²/(a+b) to enrolled histograms
```

Descriptor length: 64 cells × 59 bins = 3 776 `f32` per crop.

Matching follows the same multi-shot rule as the ArcFace `Gallery`: an identity may
enrol several crops and its score is the minimum distance to any member (one frontal
shot generalises poorly across pose). Decisions use `max_distance` (accept radius) and
an optional `min_margin` (best identity must beat the runner-up).

The LBP operator is invariant to monotone photometric changes (gain/offset) by
construction. Histogram equalisation is offered as a config option but made no
difference on this dataset (numbers below).

## 3. Evaluation methodology

The challenge with a weight-free recogniser is obtaining trustworthy identity labels
without manual annotation. The ground truth is therefore produced **offline** by the
accurate model path; the recogniser under evaluation never sees an ONNX runtime:

1. **Frames.** 90 vertical 1080×1920 frames from three short-drama clips
   (`goodshort`, `reelshort`, `vibeshort`, 30 frames each). Real broadcast content:
   subtitles, motion blur, pose and expression variation, many faceless shots.
2. **Offline labelling (`prep_lbph_crops`, `--features ort-backend` only).**
   SCRFD-10G detects the largest face per frame (score ≥ 0.5, short side ≥ 80 px);
   ArcFace R50 embeds it. Frames are grouped into identities by incremental clustering
   at cosine ≥ 0.40 — far below the measured same-identity range (min join cosine in any
   cluster was 0.51) and far above the measured different-identity cosines (≈ 0.07 on
   fixtures). The clustering audit (cluster sizes, exemplar frame names, minimum join
   cosine) is printed and written to `labels.csv` for inspection.
3. **Crop isolation.** LBPH receives a plain **bounding-box crop** (square, +30 %
   margin, bilinear 120×120, grayscale). No keypoint alignment is used: ArcFace
   landmarks never touch the evaluated path, so the number answers "how good is LBPH
   given a correct box", which is the recogniser-isolated condition.
4. **Evaluation (`bench_lbph`, default zero-dep build).** Every crop pair yields a χ²
   distance labelled same/different identity (85 same, 510 different pairs);
   leave-one-out rank-1 identification runs against the remaining crops.

Reproduce: `tools/lbph_prep.sh` (needs Pillow for JPG→PPM conversion, the two ONNX
weights, and a libonnxruntime shared library — all lab-only; the bench itself builds
with zero dependencies).

## 4. Measured results — 35 crops, 8 identities

Pair-distance distributions (raw LBP):

| pair type | n | mean | p5 | p50 | p95 | extreme |
|---|--:|--:|--:|--:|--:|--:|
| same identity | 85 | 22.3 | 10.5 | 17.8 | 53.7 | max 54.9 |
| different identity | 510 | 54.8 | 46.1 | 55.0 | 63.6 | min 32.7 |

Mean margin: **32.5** on the χ² scale.

| operating point | threshold | pair accuracy | FAR | FRR |
|---|--:|--:|--:|--:|
| **crate default** | **30** | **97.6 %** | **0.0 %** | 16.5 % |
| equal-error (EER) | 48 | — | 9.6 % | 9.4 % |

* Leave-one-out **rank-1 identification: 33/33 = 100 %** over repeated identities
  (identities with ≥ 2 crops). Two clusters contained a single crop; rank-1 cannot
  succeed for those (no same-identity gallery sample left behind) and they are reported
  separately instead of counted as errors.
* Histogram-equalised preprocessing changed nothing measurable (rank-1 33/33, same
  numbers to one decimal) — consistent with the operator's built-in monotone-lighting
  invariance.
* `DEFAULT_MAX_DISTANCE` was calibrated from this run: 30 is the empirical zero-FAR
  point on the set (closest impostor pair 32.7). It is a starting point, **not** a
  population-wide constant — enrol a handful of your own people and verify.

The full generated report is committed at `docs/bench-results-lbph.md`.

## 5. Honest limitations

* **Detection is the real zero-dep bottleneck, not recognition.** The Haar cascade
  shipped in the default build is a small hand-authored demo cascade; on the 90 drama
  frames it fired ~3 000+ windows per frame with no selectivity (every frame, including
  faceless shots, "had a face"). The LBPH numbers above use correct SCRFD boxes. A
  production zero-dep deployment must supply a real trained cascade
  (`--cascade file.rfcf`; convert the OpenCV XML with
  `tools/convert_opencv_xml.py`). Training/shipping such a cascade file is a separate
  piece of work from the recogniser and changes nothing in the LBPH accuracy envelope
  conditional on a correct box.
* **Small, single-domain dataset.** 35 accepted crops from one content type (vertically
  shot drama, faces typically near-frontal, large in frame). No age/large-pose/heavy
  occlusion splits. Treat 100 % rank-1 as "works clearly on this domain", not as a
  LFW-style claim.
* **No landmark alignment.** ArcFace normalises eyes/mouth to fixed coordinates; LBPH
  only sees the detector box. Pose and box jitter are exactly what the spatial
  histograms are sensitive to; multi-shot enrolment is the intended mitigation.
* **2004 descriptor ceiling.** LBPH under controlled frontal conditions is competitive
  for small galleries; it will not approach ArcFace under unconstrained pose/ageing.
  The measured ArcFace same/different cosine margin on fixtures is 0.93, i.e. the two
  distributions barely overlap at all. Use the `onnx` feature when the deployment
  permits it.

## 6. API sketch

```rust
use rsface::lbph::{LbphRecognizer, LbphConfig, LbphMatch};

let mut rec = LbphRecognizer::new(LbphConfig::default()); // radius 1, 8×8, 120 px, d ≤ 30
rec.enroll("alice", &alice_crop_1);
rec.enroll("alice", &alice_crop_2);

match rec.identify_crop(&probe) {
    LbphMatch::Match { label, distance, .. } => println!("{label} (d={distance:.1})"),
    LbphMatch::BelowThreshold { .. } | LbphMatch::Ambiguous { .. }
    | LbphMatch::NoCandidates => println!("unknown"),
}
rec.verify("alice", &probe); // Option<f32>: chi-square distance if the label is enrolled
```

All inputs are `GrayImage`; use `RgbImage::to_gray()` and the built-in resize if your
detector returns colour or different crop sizes.
