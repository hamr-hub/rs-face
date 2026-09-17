# Zero-dependency face recognition: Eigenfaces (PCA)

This document describes the **second** recogniser that ships in the crate's default,
zero-third-party-dependency build, and reports its measured accuracy on the same real
faces used for LBPH.

## 1. What "zero dependencies" means here

| Capability | Default build (`--no-default-features`) | Optional (`--features onnx`) |
|---|---|---|
| Detection | Viola–Jones Haar cascade (pure Rust) | SCRFD-10G (ONNX, 17 MB weights) |
| Recognition | **LBPH — no weights, no crates** ([doc](recognition-lbph.md)) | ArcFace R50 (ONNX, 174 MB weights) |
| Recognition | **Eigenfaces — no weights, no crates** (this document) | ArcFace R50 (ONNX, 174 MB weights) |

Eigenfaces ([Turk & Pentland, *Eigenfaces for Recognition*, 1991](https://doi.org/10.1162/jocn.1991.3.1.71))
learns *what this gallery's faces vary along* and describes every crop by its
coordinates in that learned subspace. Like LBPH it needs no download: "training" is one
small symmetric eigendecomposition performed at enrolment time, implemented in `std`
only (`src/eigenface.rs`) — including the Jacobi eigenvalue routine, so the default
build gains no linear-algebra crate. OpenCV's `face::EigenFaceRecognizer` implements
the same projection.

LBPH and eigenfaces are complementary classical baselines, not alternatives where one
dominates: LBPH is a local, gallery-independent descriptor (enrol a crop any time);
PCA is global and gallery-specific (the subspace moves when identities are added).
Shipping both lets a deployment compare on its own data without leaving the zero-dep
build.

## 2. Pipeline

```text
gray frame
  └─ face box (Haar in the zero-dep build; SCRFD when a model is present)
       └─ square box crop, resize to 64×64, pixels scaled to [0, 1]
            └─ subtract per-pixel gallery mean
                 └─ project onto the leading eigenfaces uᵢ
                      └─ coefficient vector cᵢ = uᵢ · (x − mean)
                           └─ L2 distance (optionally √λ-whitened) to enrolled crops
```

The naïve covariance matrix is `d × d` with `d = 64² = 4 096`; eigendecomposing it on
every enrolment would be wasteful. With `n` gallery crops we instead decompose the
`n × n` **Gram matrix** `G = X Xᵀ` (the Turk–Pentland trick) and recover the
eigenfaces from its eigenvectors:

```text
G v = λ_G v   →   u = Xᵀ v / √λ_G,    covariance eigenvalue λ = λ_G / (n − 1)
```

The symmetric eigendecomposition itself is a cyclic **Jacobi rotation** routine
(30 sweeps, tolerance `1e-10 · |λ_max|`) — no LAPACK, no third-party code.

Components are kept in descending-eigenvalue order until 98 % of eigenvalue energy is
covered (hard cap 255). Matching is nearest-neighbour over enrolled crops with the
same multi-shot rule as LBPH: an identity's score is the minimum distance to any of
its members. Two metrics are provided:

* **Euclidean** (default): plain L2 over coefficients — the classic 1991 metric.
* **Mahalanobis**: each coefficient difference is divided by √λ, so low-energy detail
  directions weigh as much as the dominant (often lighting-driven) ones.

## 3. Evaluation methodology — strict leave-one-out

The crop set and ground truth are identical to the LBPH evaluation: **35 crops / 8
identities** in `out/lbph/crops`, labels from offline ArcFace clustering; the evaluated
path never touches ONNX. 85 same-identity and 510 different-identity unordered pairs.
See [`recognition-lbph.md` §3](recognition-lbph.md) for how the crops and labels are
produced (`prep_lbph_crops`, lab-only, `--features ort-backend`).

The crucial protocol difference from LBPH is that the PCA subspace is **gallery
specific**, so holding a single model fixed would leak every probe into the training
set. `bench_eigenface` therefore does strict leave-one-out: for each probe it
**retrains the recogniser on the other n−1 crops**, projects the probe, and scores it
against the retrained gallery.

Two measurements come out of that loop, with different evidential weight:

* **LOO rank-1 identification — the primary metric.** The probe is never in the model
  that ranks it, so this is honest.
* **Pair FAR/FRR.** One directed distance per unordered pair; the gallery endpoint of
  each distance *is* a training point of the model used, so the reported FAR/FRR is
  **mildly optimistic** relative to unseen data. We report it anyway for comparability
  with the LBPH numbers, but threshold calibration should lean conservative and be
  re-checked on the target deployment.

Singletons (identities with one crop, here `id01` and `id07`) cannot have a
same-identity gallery member left behind and are counted separately, not as errors.

Reproduce: `tools/lbph_prep.sh` (crop preparation) then
`cargo run --release --bin bench_eigenface -- out/lbph/crops`.

## 4. Measured results — 35 crops, 8 identities

All four preprocessing/metric variants, strict LOO:

| variant | margin¹ | best-threshold pair acc | EER threshold | LOO rank-1 |
|---|--:|--:|--:|--:|
| **Euclidean, raw (crate default)** | **20.80** | **97.3 %** @ 6.03 | **6.33** | **33/33 = 100 %** |
| Euclidean, histogram-equalised | 17.19 | 98.0 % @ 13.80 | 14.87 | 33/33 = 100 % |
| Mahalanobis, raw | 2.46 | 94.8 % @ 2.98 | 3.78 | 33/33 = 100 % |
| Mahalanobis, equalised | 1.79 | 91.8 % | 4.91 | **29/33 = 87.9 %** |

¹ margin = mean different-identity distance − mean same-identity distance.

Distance distributions for the shipped variant (Euclidean, raw):

| pair type | n | mean | p5 | p50 | p95 | extreme |
|---|--:|--:|--:|--:|--:|--:|
| same identity | 85 | 2.70 | 0.43 | 1.94 | 6.06 | max 7.40 |
| different identity | 510 | 23.49 | 6.92 | 25.52 | 31.55 | min 3.90 |

| operating point | threshold | pair accuracy | FAR | FRR |
|---|--:|--:|--:|--:|
| **crate default** `DEFAULT_MAX_DISTANCE` | **6.3** | **97.0 %** | **2.9 %** | **3.5 %** |
| best pair accuracy | 6.03 | 97.3 % | 2.2 % | 5.9 % |
| equal-error (EER) | 6.33 | — | 2.9 % | 2.4 % |

Calibration notes, stated plainly:

* Unlike LBPH on this set, the distributions **overlap**: the closest impostor pair
  (3.90) is well inside the genuine range (max 7.40). There is **no zero-FAR threshold
  that still accepts a useful fraction of probes** — a threshold at 3.9 buys FAR 0 % at
  the cost of FRR ≈ 33 %. The shipped default 6.3 is therefore the EER-region point
  (FAR ≈ FRR ≈ 3 %), not a zero-FAR point.
* Whitening by √λ (Mahalanobis) collapses the margin from 20.8 to 2.5: on this
  near-frontal, similarly-lit material the dominant components already carry the
  identity signal, and amplifying low-energy noise directions only adds impostors. It
  is kept as an option for flatter-spectrum galleries but is not the default.
* Histogram equalisation slightly improves the *best-threshold* pair accuracy (98.0 %)
  but at double the threshold scale and with a smaller margin; rank-1 is unchanged.
  Raw pixels win on margin and simplicity.

The full generated report is committed at `docs/bench-results-eigenface.md`.

## 5. Honest limitations

* **Global, gallery-specific model.** A single occluding bar or a cast shadow moves
  *every* coefficient; the subspace itself is computed from the enrolled crops. The
  model **must be retrained when identities change** — there is deliberately no
  incremental `enroll` like LBPH has; call `EigenfaceRecognizer::train` again with
  the new crop list. Enrolment is cheap for small galleries (the decomposition is
  `O(n³)` with `n ≪ d`), not for thousands of identities.
* **Threshold scale is not portable.** The coefficient L2 scale depends on crop size,
  retained energy, and gallery composition. `DEFAULT_MAX_DISTANCE = 6.3` is calibrated
  on 35 drama crops; rerun `bench_eigenface` on your own data before trusting it.
* **Same detection bottleneck as LBPH.** The numbers answer "how good are eigenfaces
  given a correct box". The built-in Haar cascade is a demo and is not selective
  enough for production; supply a trained `.rfcf` cascade or use SCRFD. See
  [`recognition-lbph.md` §5](recognition-lbph.md).
* **Small, single-domain dataset and no landmark alignment.** 35 near-frontal crops
  from one content type; eyes/mouth are not normalised to fixed coordinates (only the
  detector box is). Treat 100 % LOO rank-1 as "clearly works on this domain", not an
  LFW claim. Pose/ageing/occlusion behaviour degrades faster for global PCA than for
  local descriptors — that is the classical result, and the Mahalanobis row is a small
  visible taste of it.
* **Classical ceiling.** For uncontrolled scenes the `ort-backend`/`tract-backend`
  ArcFace path remains the production recogniser; eigenfaces are a zero-dep baseline
  and a teaching/diagnostic tool that also works fully offline.

## 6. API sketch

```rust
use rsface::eigenface::{EigenfaceRecognizer, EigenfaceConfig, EigenMatch};

// One-shot training: the subspace is learned from the whole gallery.
let samples = [
    ("alice", &alice_crop_1),
    ("alice", &alice_crop_2),
    ("bob",   &bob_crop_1),
];
let rec = EigenfaceRecognizer::train(EigenfaceConfig::default(), samples)?;
//       face_size 64, 98% energy, cap 255, Euclidean, max_distance 6.3

match rec.identify_crop(&probe) {
    EigenMatch::Match { label, distance, .. } => println!("{label} (d={distance:.2})"),
    EigenMatch::BelowThreshold { best } | EigenMatch::Ambiguous { .. }
    | EigenMatch::NoCandidates => println!("unknown"),
}
rec.verify("alice", &probe); // Option<f32>: distance if "alice" is enrolled

// Reuse one projection for several queries:
let c = rec.project(&probe);
let ranking: Vec<(String, f32)> = rec.rank(&c);
```

Adding an identity means retraining: rebuild the sample list and call `train` again.
All inputs are `GrayImage`; use `RgbImage::to_gray()` for colour detector output.
