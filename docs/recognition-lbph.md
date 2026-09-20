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
`std` plus image primitives already in `src/image/`. The sampler, the uniform-pattern
lookup table, spatial-cell geometry, and the chi-square metric match OpenCV's
`face::LBPHFaceRecognizer` bit-for-bit (`elbp_`: bit 0 at the 3 o'clock sample,
neighbour `n` at `(r·cos 2πn/P, −r·sin 2πn/P)`, bilinear interpolation at every
radius, epsilon-tie comparison, interior-only, fixed `interior/grid` cell rectangles;
8 neighbours, 59-bin uniform mapping). One deliberate deviation: the spatial grid is
**6×6 cells rather than OpenCV's 8×8** — under the exact sampler both land at 60/68
LOO rank-1 on the hard gallery (§4.2) and 6×6 keeps the descriptor compact; 8×8 and
10×10 remain one config field away.

> Numbers in §4 are measured with this OpenCV-exact sampler and sorted-path bench
> order. Earlier revisions of this document used a nearest-pixel 6 o'clock convention
> and unsorted directory iteration; their results are **not comparable** across the
> change (the gallery format was bumped from `RSLB` v1 to v2 for the same reason).

## 2. Pipeline

```text
gray frame
  └─ face box (Haar in the zero-dep build; SCRFD when a model is present)
       └─ square box crop, +30 % margin, resize to 120×120
            └─ LBP codes (8 neighbours on the r-ring, bilinear-sampled,
                 OpenCV elbp convention: bit 0 at 3 o'clock, ≥-with-eps tie)
                 └─ map to 59 uniform bins
                      └─ 6×6 spatial cell histograms (fixed interior/grid
                         rectangles, edge remainder dropped), L1 per cell
                           └─ χ² distance Σ(a−b)²/(a+b) to enrolled histograms
```

Descriptor length: 36 cells × 59 bins = 2 124 `f32` per crop.

Matching follows the same multi-shot rule as the ArcFace `Gallery`: an identity may
enrol several crops and its score is the minimum distance to any member (one frontal
shot generalises poorly across pose). Decisions use `max_distance` (accept radius) and
an optional `min_margin` (best identity must beat the runner-up).

The LBP operator is invariant to monotone photometric changes (gain/offset) by
construction. Histogram equalisation is offered as a config option but loses on both
current evaluation sets (§4), so it stays opt-in.

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
   configurations rank by LOO rank-1, then margin. Crops are visited in sorted
   path order so exact distance ties resolve deterministically between runs.

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
| same identity | 195 | 15.92 | 4.85 | 17.31 | 29.22 | max 37.92 |
| different identity | 2 731 | 26.41 | 21.08 | 26.03 | 33.10 | min 17.50 |

Mean margin: **10.49** on the χ² scale, and the tails overlap (the closest impostor,
17.50, sits below the farthest genuine pair, 37.92). Leave-one-out **rank-1
identification is 60/68 = 88.2 %** over repeated identities.

| operating point | threshold | pair accuracy | FAR | FRR |
|---|--:|--:|--:|--:|
| **crate default** `DEFAULT_MAX_DISTANCE` | **16.7** | **96.6 %** | **0.00 %** | **50.8 %** |
| best pair accuracy | 17.86 | 96.8 % | 0.07 % | 47.7 % |
| equal-error (EER) | 23.21 | — | 19.7 % | 19.5 % |

Histogram equalisation hurts slightly here (56/68 rank-1, EER 23.04, FAR 0.04 % at
the default) — consistent enough with the operator's built-in monotone-lighting
invariance that the raw variant stays the default; benchmark both on your own data.

The full generated report (including per-identity crop counts) is committed at
`docs/bench-results-lbph.md`.

### 4.2 Hyperparameter sweep — why the default grid stays 6×6

LOO rank-1 over the 68 repeated-identity probes, raw LBP (sorted-path order):

| grid | 90 px | 120 px | 150 px |
|---|--:|--:|--:|
| **6×6 (shipped)** | 60/68 | **60/68** | 60/68 |
| 8×8 (OpenCV default) | 59/68 | 60/68 | 60/68 |
| 10×10 | 57/68 | 59/68 | 60/68 |

Under the OpenCV-exact sampler the three grids are statistically tied on 68 probes:
the best 8×8 and 10×10 cells reach 60/68, the same as 6×6, and the bench's
margin tiebreak names 10×10/150 (pair-distance margin 35.4) the formal winner. 6×6 at
120 px stays the shipped default because it sits on that rank-1 Pareto front with a
smaller descriptor (2 124 bins vs 3 776 at 8×8, 5 900 at 10×10); claiming a multi-probe
accuracy edge for any cell here would be noise. On the easy 35-crop gallery finer
grids do pick up one probe (6×6: 32/33 at every crop size; 8×8: 32/33→33/33 from
120 px; 10×10: 33/33 throughout). Deployments feeding landmark-aligned crops may
prefer 8×8/10×10 for the extra spatial precision.

### 4.3 Easy gallery (no-regression set) — 35 crops, 8 identities

The original gallery stays near-frontal and well lit; the shipped config reaches
32/33 rank-1 (97.0 %) and finer grids reach 33/33. Same-identity distances stay low
(mean 10.3, p95 29.2) while every impostor pair is ≥ 17.9, so at the shipped
constant 16.7 the pair operating point is **FAR 0.0 %, FRR 17.6 %** (85 same / 510
different pairs; EER 22.95, FAR/FRR ≈ 12.9 %): the same conservative threshold that
suppresses false accepts on the hard set works here as well.

### 4.4 Threshold policy

False accepts are the expensive error in access-control-style verification, so
`DEFAULT_MAX_DISTANCE = 16.7` ships a **deliberately conservative low-FAR point**, not
the EER: it sits just below the closest measured impostor distance on both sets
(17.5 hard / 17.9 easy), so the measured false-accept count is 0/2 731 and 0/510
while it accepts roughly half of genuine probes on the hard set (FRR 50.8 %);
uncertain probes surface as `LbphMatch::BelowThreshold`. The distributions overlap
there — *no* threshold gives both low FAR and low FRR (EER ≈ 20 %) — so for the
close-set case ("which enrolled person is this?"), prefer the threshold-free
`LbphRecognizer::rank_crop` API, whose 88.2 % rank-1 (60/68) is the honest headline
on hard data. The χ² scale is descriptor-specific
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
  cross-title identity overlap. Treat 88.2 % rank-1 as "works on this harder domain",
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

// Persist descriptors (not crops) and rebuild later in a new process:
rec.save("gallery.lbph")?;                 // atomic sibling-temp + rename
let rec = LbphRecognizer::load("gallery.lbph")?; // bit-exact, fully validated
```

The gallery file format (`RSLB` v2, zero-dependency little-endian, ~8.5 KB per
crop at the 6×6 default; v1 was a pre-release build with a different sampling
convention and is rejected) is specified in
[`gallery-persistence.md`](gallery-persistence.md).

All inputs are `GrayImage`; use `RgbImage::to_gray()` and the built-in resize if your
detector returns colour or different crop sizes.
