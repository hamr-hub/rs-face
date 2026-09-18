# Zero-dependency face recognition: LBPH

This document describes the recogniser that ships in the crate's **default,
zero-third-party-dependency** build and reports its measured accuracy on real faces.

## 1. What "zero dependencies" means here

| Capability | Default build (`--no-default-features`) | Optional (`--features ort-backend`) |
|---|---|---|
| Detection | Viola–Jones Haar cascade (pure Rust) | SCRFD-10G (ONNX, 17 MB weights) |
| Recognition | **LBPH — no weights, no crates** | ArcFace R50 (ONNX, 174 MB weights) |

LBPH ([Ahonen, Hadid, Pietikäinen, ECCV 2004](https://link.springer.com/chapter/10.1007/978-3-540-24670-1_36))
needs no training-time download: the "model" is the set of per-identity histograms
collected at enrolment. The implementation lives in `src/lbph.rs` and uses only
`std` plus image primitives already in `src/image/`. The descriptor, the uniform-pattern
lookup table, and the chi-square metric follow OpenCV's `face::LBPHFaceRecognizer`
(radius 1, 8 neighbours, 59-bin uniform mapping, chi-square distance). One deliberate
deviation: the spatial grid is **6×6 cells rather than OpenCV's 8×8**, chosen by a
measured sweep (§4.2); 8×8 and 10×10 remain one config field away.

## 2. Pipeline

```text
gray frame
  └─ face box (Haar in the zero-dep build; SCRFD when a model is present)
       └─ square box crop, +30 % margin, resize to 120×120
            └─ LBP codes (3×3, 8 neighbours, threshold vs. centre)
                 └─ map to 59 uniform bins
                      └─ 6×6 spatial cell histograms, L1-normalised per cell
                           └─ χ² distance Σ(a−b)²/(a+b) to enrolled histograms
```

Descriptor length: 36 cells × 59 bins = 2 124 `f32` per crop.

Matching follows the same multi-shot rule as the ArcFace `Gallery`: an identity may
enrol several crops and its score is the minimum distance to any member (one frontal
shot generalises poorly across pose). Decisions use `max_distance` (accept radius) and
an optional `min_margin` (best identity must beat the runner-up).

The LBP operator is invariant to monotone photometric changes (gain/offset) by
construction. Histogram equalisation is offered as a config option but made no
difference on either evaluation set (numbers below).

## 3. Evaluation methodology

The challenge with a weight-free recogniser is obtaining trustworthy identity labels
without manual annotation. The ground truth is therefore produced **offline** by the
accurate model path; the recogniser under evaluation never sees an ONNX runtime:

1. **Frames.** 150 vertical 1080×1920 frames, 30 each from five short-drama clips
   (`dramabox`, `goodshort`, `reelshort`, `shorttv`, `vibeshort`). Real broadcast
   content: subtitles, motion blur, pose and expression variation, many faceless shots.
2. **Offline labelling (`prep_lbph_crops`, `--features ort-backend` only).**
   SCRFD-10G detects the largest face per frame (score ≥ 0.5, short side ≥ 80 px);
   ArcFace R50 embeds it. Frames are grouped into identities by incremental clustering
   at cosine ≥ 0.40 — below the measured same-identity range (the minimum join cosine
   in any cluster was 0.476) and far above the measured different-identity cosines
   (≈ 0.07 on fixtures). The clustering audit (cluster sizes, exemplar frame names,
   minimum join cosine) is printed and written to `labels.csv` for inspection. Clips
   whose join cosines sat in the 0.48–0.55 grey zone were additionally audited
   visually with per-identity contact sheets: every cluster is the same actor under
   pose/lighting variation, no mismatch was accepted.
3. **Crop isolation.** LBPH receives a plain **bounding-box crop** (square, +30 %
   margin, bilinear 120×120, grayscale). No keypoint alignment is used: ArcFace
   landmarks never touch the evaluated path, so the number answers "how good is LBPH
   given a correct box", which is the recogniser-isolated condition.
4. **Evaluation (`bench_lbph`, default zero-dep build).** Descriptors are
   gallery-independent, so every crop is described once; every unordered crop pair
   yields a χ² distance labelled same/different identity, and a leave-one-out rank-1
   loop runs against the remaining crops. The bench also sweeps grid size
   (6×6/8×8/10×10), crop size (90/120/150 px), and histogram equalisation;
   configurations rank by LOO rank-1, then margin.

77 of the 150 frames pass the face gate and form the current **hard gallery: 21
identities, 195 same-identity and 2 731 different-identity pairs**. Nine identities
are singletons; their nine probes are reported separately (rank-1 cannot succeed with
no same-identity gallery member left behind), leaving **68 repeated-identity probes**.

The earlier, easier gallery — the 35 accepted crops from the three original clips
(`goodshort` + `reelshort` + `vibeshort`, 8 identities, 85 same / 510 different
pairs) — is kept in the lab tree (`out/lbph/crops_old`) and re-evaluated on every
config change as a **no-regression set**.

Reproduce: `tools/lbph_prep.sh` (needs Pillow for JPG→PPM conversion, the two ONNX
weights, and a libonnxruntime shared library — all lab-only; the bench itself builds
with zero dependencies).

## 4. Measured results

### 4.1 Hard gallery — 77 crops, 21 identities

Pair-distance distributions for the shipped configuration (raw LBP, 6×6, 120 px):

| pair type | n | mean | p5 | p50 | p95 | extreme |
|---|--:|--:|--:|--:|--:|--:|
| same identity | 195 | 15.14 | 4.73 | 15.77 | 26.36 | max 37.74 |
| different identity | 2 731 | 25.53 | 20.50 | 25.33 | 31.51 | min 13.18 |

Mean margin: **10.39** on the χ² scale, and the tails overlap (the closest impostor,
13.18, sits below the farthest genuine pair, 37.74). Leave-one-out **rank-1
identification is 62/68 = 91.2 %** over repeated identities.

| operating point | threshold | pair accuracy | FAR | FRR |
|---|--:|--:|--:|--:|
| **crate default** `DEFAULT_MAX_DISTANCE` | **16.7** | **96.5 %** | **0.26 %** | **48.7 %** |
| best pair accuracy | 16.68 | 96.5 % | 0.22 % | 48.7 % |
| equal-error (EER) | 22.47 | — | 20.0 % | 20.0 % |

Histogram equalisation changes nothing of substance (62/68 rank-1, EER 22.24) —
consistent with the operator's built-in monotone-lighting invariance.

The full generated report (including per-identity crop counts) is committed at
`docs/bench-results-lbph.md`.

### 4.2 Hyperparameter sweep — why the default grid is 6×6

LOO rank-1 over the 68 repeated-identity probes, raw LBP:

| grid | 90 px | 120 px | 150 px |
|---|--:|--:|--:|
| **6×6 (shipped)** | 61/68 | **62/68** | 61/68 |
| 8×8 (OpenCV default) | 59/68 | 59/68 | 60/68 |
| 10×10 | 59/68 | 60/68 | 59/68 |

The coarser 6×6 grid beats OpenCV's 8×8 at **every** crop size, by 2–3 probes: larger
cells pool the box-crop localisation jitter that an unaligned pipeline (no eye
positions, detector box only) inevitably produces. The same configs on the easy
35-crop gallery are all 33/33 — the change costs nothing there. Deployments feeding
landmark-aligned crops can set 8×8 or 10×10 and gain spatial precision instead.

### 4.3 Easy gallery (no-regression set) — 35 crops, 8 identities

The original gallery stays near-frontal and well lit; every swept configuration
reaches 33/33 rank-1. At the shipped constant 16.7 the pair operating point is
**FAR 1.2 %, FRR 16.5 %** (85 same / 510 different pairs) — the same conservative
threshold that nearly suppresses false accepts on the hard set is usable there too.

### 4.4 Threshold policy

False accepts are the expensive error in access-control-style verification, so
`DEFAULT_MAX_DISTANCE = 16.7` ships a **deliberately conservative low-FAR point**, not
the EER: on the hard gallery it accepts roughly half of genuine probes while admitting
under 0.3 % of impostors, and uncertain probes surface as
`LbphMatch::BelowThreshold`. The distributions overlap there — *no* threshold gives
both low FAR and low FRR — so for the close-set case ("which enrolled person is
this?"), prefer the threshold-free `LbphRecognizer::rank_crop` API, whose 91.2 %
rank-1 is the honest headline on hard data. The χ² scale is descriptor-specific
(grid × crop size); rerun `bench_lbph` on your own crops before trusting the constant.
(`f32::MAX` would be OpenCV's "always identify" default — useless for verification.)

## 5. Honest limitations

* **Detection is the real zero-dep bottleneck, not recognition.** The Haar cascade
  shipped in the default build is a small hand-authored demo cascade; on the drama
  frames it fired ~3 000+ windows per frame with no selectivity (every frame, including
  faceless shots, "had a face"). The numbers above use correct SCRFD boxes. A
  production zero-dep deployment must supply a real trained cascade
  (`--cascade file.rfcf`; convert the OpenCV XML with
  `tools/convert_opencv_xml.py`). Training/shipping such a cascade file is a separate
  piece of work from the recogniser and changes nothing in the LBPH accuracy envelope
  conditional on a correct box.
* **Small, clustered dataset.** 77 accepted crops across 21 identities from one
  content type (vertically shot drama); identities are uneven (1–11 crops each) and
  nine are singletons. There is no age/large-pose/heavy-occlusion split and no
  cross-title identity overlap. Treat 91.2 % rank-1 as "works on this harder domain",
  not as an LFW-style claim.
* **No landmark alignment.** ArcFace normalises eyes/mouth to fixed coordinates; LBPH
  only sees the detector box. Pose and box jitter are exactly what the spatial
  histograms are sensitive to; the 6×6 default pools part of that jitter, and
  multi-shot enrolment is the intended mitigation.
* **Pair FAR/FRR is mildly optimistic.** Each descriptor is a gallery member when
  scored against every other crop, so same-pair distances benefit slightly from
  enrolment-time representation. Rank-1 LOO (probe never enrolled against itself) does
  not have this leak and is the primary metric.
* **2004 descriptor ceiling.** Even tuned, LBPH under unconstrained pose/ageing will
  not approach ArcFace; the measured ArcFace same/different cosine margin on fixtures
  is 0.93, i.e. those distributions barely overlap. Use the `ort-backend`/
  `tract-backend` path when the deployment permits it.

## 6. API sketch

```rust
use rsface::lbph::{LbphRecognizer, LbphConfig, LbphMatch};

let mut rec = LbphRecognizer::new(LbphConfig::default()); // radius 1, 6×6, 120 px, d ≤ 16.7
rec.enroll("alice", &alice_crop_1);
rec.enroll("alice", &alice_crop_2);

// Close-set identification: threshold-free ranking, recommended when the probe is
// known to be one of the enrolled identities.
let ranking = rec.rank_crop(&probe); // Vec<(String, f32)>, best first

// Verification with the conservative low-FAR accept radius:
match rec.identify_crop(&probe) {
    LbphMatch::Match { label, distance, .. } => println!("{label} (d={distance:.1})"),
    LbphMatch::BelowThreshold { .. } | LbphMatch::Ambiguous { .. }
    | LbphMatch::NoCandidates => println!("unknown"),
}
rec.verify("alice", &probe); // Option<f32>: chi-square distance if the label is enrolled
```

All inputs are `GrayImage`; use `RgbImage::to_gray()` and the built-in resize if your
detector returns colour or different crop sizes.
