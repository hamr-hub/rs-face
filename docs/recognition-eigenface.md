# Zero-dependency face recognition: Eigenfaces (PCA)

This document describes the **second** recogniser that ships in the crate's default,
zero-third-party-dependency build, and reports its measured accuracy on the same real
faces used for LBPH.

## 1. What "zero dependencies" means here

| Capability | Default build (`--no-default-features`) | Optional (`--features ort-backend`) |
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

The crop set and ground truth are identical to the LBPH evaluation: **77 crops / 21
identities** in `out/lbph/crops` (150 frames from five short-drama clips, labels from
offline ArcFace clustering; the evaluated path never touches ONNX). 195 same-identity
and 2 731 different-identity unordered pairs. See
[`recognition-lbph.md` §3](recognition-lbph.md) for the frame provenance, clustering
and visual-label audit, and the no-regression role of the earlier 35-crop /
8-identity easy gallery (`out/lbph/crops_old`).

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

Nine of the 21 identities are singletons (one accepted crop each); those nine probes
cannot have a same-identity gallery member left behind and are counted separately, not
as errors, leaving 68 repeated-identity probes.

The bench additionally sweeps crop size (32/48/64 px) and retained eigenvalue energy
(90/95/98/100 %) for the Euclidean/raw variant; rows rank by strict-LOO rank-1 and
then margin. Reproduce: `tools/lbph_prep.sh` (crop preparation) then
`cargo run --release --bin bench_eigenface -- out/lbph/crops`.

## 4. Measured results

### 4.1 Hard gallery — 77 crops, 21 identities

All four preprocessing/metric variants at the shipped 64 px / 98 %-energy defaults,
strict LOO:

| variant | margin¹ | best-threshold pair acc | EER threshold | LOO rank-1 |
|---|--:|--:|--:|--:|
| **Euclidean, raw (crate default)** | **11.92** | **95.8 %** @ 5.72 | **14.04** | **58/68 = 85.3 %** |
| Euclidean, histogram-equalised | 10.33 | 96.5 % | 20.33 | 56/68 = 82.4 % |
| Mahalanobis, raw | 2.01 | 95.4 % | 7.11 | 43/68 = 63.2 % |
| Mahalanobis, equalised | 1.84 | 95.3 % | 8.15 | 38/68 = 55.9 % |

¹ margin = mean different-identity distance − mean same-identity distance.

Distance distributions for the shipped variant (Euclidean, raw):

| pair type | n | mean | p5 | p50 | p95 | extreme |
|---|--:|--:|--:|--:|--:|--:|
| same identity | 195 | 8.14 | 0.85 | 6.38 | 19.48 | max 22.31 |
| different identity | 2 731 | 20.06 | 8.14 | 20.11 | 30.82 | min 3.78 |

| operating point | threshold | pair accuracy | FAR | FRR |
|---|--:|--:|--:|--:|
| **crate default** `DEFAULT_MAX_DISTANCE` | **6.3** | **95.1 %** | **1.5 %** | **51.8 %** |
| best pair accuracy | 5.72 | 95.8 % | 0.6 % | 54.9 % |
| equal-error (EER) | 14.04 | — | 20.5 % | 20.5 % |

The same/different distributions overlap heavily (the closest impostor, 3.78, is well
inside the genuine range), so as with LBPH the shipped constant is a **conservative
low-FAR point**, not an EER point: it admits under 1.6 % of impostors at the cost of
rejecting about half of genuine probes. For the close-set question ("which enrolled
person is this?") use the threshold-free `EigenfaceRecognizer::rank` / `rank_crop`
ranking — strict-LOO rank-1 is the honest headline at **85.3 %**.

Calibration notes, stated plainly:

* Whitening by √λ (Mahalanobis) collapses the margin from 11.9 to 2.0 and rank-1 from
  58/68 to 43/68: on this material the dominant components already carry the identity
  signal, and amplifying low-energy noise directions only adds impostors. It is kept as
  an option for flatter-spectrum galleries but is not the default.
* Histogram equalisation improves the *best-threshold* pair accuracy slightly (96.5 %)
  but at a larger threshold scale, with a smaller margin and two fewer rank-1 probes.
  Raw pixels win on margin, rank-1, and simplicity.

### 4.2 Hyperparameter sweep — the defaults already sit on the PCA ceiling

Strict-LOO rank-1 for Euclidean/raw across crop size and retained eigenvalue energy:

| crop size | 90 % | 95 % | 98 % (shipped) | 100 % |
|---|--:|--:|--:|--:|
| 32 px | 56/68 | 57/68 | 58/68 | 57/68 |
| 48 px | 57/68 | 58/68 | 58/68 | 57/68 |
| **64 px (shipped)** | 57/68 | **58/68** | **58/68** | 57/68 |

No configuration beats **58/68 = 85.3 %**: that is the classical-PCA ceiling on this
gallery, and the shipped 64 px / 98 % point sits on it. Five rows tie at the ceiling;
the tie-break names 64 px / 95 % (margin 12.28 vs 11.92), but the difference is within
one probe of sampling noise and the accept constant is calibrated in the shipped
coefficient scale, so the 98 % default is kept. Throwing away the last 2 % of energy or
shrinking the crop cannot manufacture identity signal that global linear projection
does not contain.

The full generated report (all 16 rows, distributions, per-identity counts) is
committed at `docs/bench-results-eigenface.md`.

### 4.3 Easy gallery (no-regression set) — 35 crops, 8 identities

On the original near-frontal gallery the shipped variant still achieves 33/33 strict
LOO rank-1, and 6.3 sits at the EER there rather than below it: pair accuracy 97.0 %
with FAR 2.9 % / FRR 3.5 % (closest impostor 3.90 vs farthest genuine 7.40, EER
threshold ≈ 6.33). One constant therefore gives an EER-region operating point on easy
data and a conservative low-FAR point on hard data — exactly the direction a
verification default should fail.

## 5. Honest limitations

* **Global, gallery-specific model.** A single occluding bar or a cast shadow moves
  *every* coefficient; the subspace itself is computed from the enrolled crops. The
  model **must be retrained when identities change** — there is deliberately no
  incremental `enroll` like LBPH has; call `EigenfaceRecognizer::train` again with
  the new crop list. Enrolment is cheap for small galleries (the decomposition is
  `O(n³)` with `n ≪ d`), not for thousands of identities.
* **Threshold scale is not portable.** The coefficient L2 scale depends on crop size,
  retained energy, and gallery composition. `DEFAULT_MAX_DISTANCE = 6.3` is calibrated
  on the repo's two drama galleries (EER-region on the easy 35-crop set, conservative
  low-FAR on the hard 77-crop set); rerun `bench_eigenface` on your own data before
  trusting it.
* **Same detection bottleneck as LBPH.** The numbers answer "how good are eigenfaces
  given a correct box". The built-in Haar cascade is a demo and is not selective
  enough for production; supply a trained `.rfcf` cascade or use SCRFD. See
  [`recognition-lbph.md` §5](recognition-lbph.md).
* **Small, clustered dataset and no landmark alignment.** 77 crops / 21 unevenly
  populated identities from one content type; eyes/mouth are not normalised to fixed
  coordinates (only the detector box is). Treat 85.3 % LOO rank-1 as "works on this
  harder domain", not an LFW claim — and note the easy-gallery 100 % did not survive
  harder pose and lighting: global PCA degrades faster than local descriptors, the
  classical result, and the Mahalanobis collapse (63.2 % → 55.9 %) is a small visible
  taste of how brittle global coefficient scaling is.
* **Classical ceiling.** The 16-point sweep in §4.2 shows the PCA ceiling on this
  gallery is 58/68 and the defaults already reach it; no configuration tuning of crop
  size or retained energy closes the gap to LBPH, let alone ArcFace. For uncontrolled
  scenes the `ort-backend`/`tract-backend` ArcFace path remains the production
  recogniser; eigenfaces are a zero-dep baseline and a teaching/diagnostic tool that
  also works fully offline.

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
