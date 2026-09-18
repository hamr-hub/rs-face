# Zero-dependency face recognition: Fisherfaces (LDA)

This document describes the **third** recogniser that ships in the crate's default,
zero-third-party-dependency build, and reports its measured accuracy on the same two
real-face galleries used for LBPH and eigenfaces.

## 1. Why a third classical recogniser

| Capability | Default build | Optional (`--features ort-backend`) |
|---|---|---|
| Recognition | **LBPH** — local, gallery-independent descriptor ([doc](recognition-lbph.md)) | ArcFace R50 (ONNX, 174 MB weights) |
| Recognition | **Eigenfaces** — PCA, maximises total scatter ([doc](recognition-eigenface.md)) | ArcFace R50 |
| Recognition | **Fisherfaces** — LDA, maximises between-class separation (this document) | ArcFace R50 |

Eigenfaces answers "along what directions do these crops vary?", and the answer on
drama-frame crops is dominated by lighting and pose. Fisherfaces
([Belhumeur, Hespanha, Kriegman, *Eigenfaces vs. Fisherfaces*, PAMI 1997](https://doi.org/10.1109/34.598228))
answers a sharper question: "along what directions do the *enrolled identities*
separate while each identity's crops stay together?" — Fisher's linear discriminant.
It is the natural discriminative counterpart to PCA and, like the other two, needs no
download and no third-party crate (`src/fisherface.rs`, Jacobi eigensolver shared in
`src/linalg.rs`). OpenCV's `face::FisherFaceRecognizer` implements the same projection.

The hypothesis under test here was specific: eigenfaces hit a sweep-proven **58/68
rank-1 ceiling** on the hard gallery (see
[`recognition-eigenface.md` §4.2](recognition-eigenface.md)). Can a class-aware linear
projection break it? Measured answer: by one probe — 59/68. It does not catch LBPH
(62/68), but it gives the **best pair-distance EER of the three** zero-dep
recognisers (§4.1), which makes it the strongest classical *verification* option.

## 2. Pipeline and math

```text
gray frame
  └─ face box (Haar in the zero-dep build; SCRFD when a model is present)
       └─ square box crop, resize to 64×64, pixels scaled to [0, 1]
            └─ subtract per-pixel gallery mean m
                 └─ PCA reduction: keep the leading n−C total-scatter directions
                      └─ build S_W (within-class) and S_B (between-class) in PCA space
                           └─ whiten S_W, eigendecompose S_W^{-1/2} S_B S_W^{-1/2}
                                └─ keep at most C−1 Fisher axes, map back to pixels
                                     └─ Euclidean NN on the ≤ C−1 projection coefficients
```

With `n` crops, `C` identities and `d = 64² = 4 096` pixels:

1. **PCA reduction.** `S_W` is singular in `d`-space (`d ≫ n`) and has null space in
   every direction not spanned by the crops. The textbook fix keeps exactly the
   `n − C` leading principal directions of the total scatter (Gram-matrix trick plus
   the shared cyclic-Jacobi routine, 30 sweeps, tolerance `1e-10 · |λ_max|`). This
   discards the null space in which within-class covariance is undefined while
   retaining all class-discriminative rank.
2. **Scatter matrices.** In the `p = n − C` dimensional PCA space:

   ```text
   S_W = Σ_k Σ_{i∈k} (z_i − μ_k)(z_i − μ_k)ᵀ      (rank-1 differences to class means)
   S_B = Σ_k n_k μ_k μ_kᵀ
   ```

3. **Generalised eigenproblem.** `S_B w = λ S_W w` is made symmetric with the
   whitening `S_W^{−1/2} S_B S_W^{−1/2}`, solved with the same Jacobi routine. Real
   galleries with near-duplicate frames still leave `S_W` rank-deficient inside the
   PCA space, so eigenvalues below `1e-9 · λ_max(S_W)` use a pseudo-inverse weight
   (their whitening columns are zeroed) instead of an inverted noise direction.
4. **Fisherfaces.** At most `C − 1` eigenvalues are positive; those eigenvectors are
   transformed back through the whitening and PCA bases into pixel space and
   unit-normalised. The stored descriptor of every crop is therefore at most
   `C − 1` coefficients long — 20 `f32`s on a 21-identity gallery, vs 2 124 for LBPH.
5. **Match.** Euclidean nearest neighbour with the same multi-shot rule as the other
   recognisers: an identity's score is the minimum coefficient distance to any of its
   crops.

Histogram equalisation is available as `FisherfaceConfig { equalize: true, .. }`, and
`max_components` can truncate below `C − 1`; neither is the default for the reasons in
§4.1–4.2.

## 3. Evaluation methodology — strict leave-one-out

Crops, labels, and protocol are identical to the eigenface evaluation: **77 crops / 21
identities** in `out/lbph/crops` (150 frames from five short-drama clips, labels from
offline ArcFace clustering; the evaluated path never touches ONNX), 195 same-identity
and 2 731 different-identity unordered pairs; nine singleton identities leave **68
repeated-identity probes** for rank-1. The earlier 35-crop / 8-identity set in
`out/lbph/crops_old` is the easy no-regression gallery. Provenance and the visual-label
audit are in [`recognition-lbph.md` §3](recognition-lbph.md).

Because the discriminant subspace is gallery specific, `bench_fisherface` does strict
leave-one-out: for each probe it **retrains PCA→LDA on the other n−1 crops**, projects
the probe, and scores it against the retrained gallery.

* **LOO rank-1 identification — the primary metric.** The probe is never in the
  model that ranks it.
* **Pair FAR/FRR.** One directed distance per unordered pair; the gallery endpoint is
  a training point of the model used, so pair FAR/FRR is **mildly optimistic** versus
  unseen data. Thresholds are calibrated conservatively.

The bench scores the two canonical preprocessing variants at the shipped 64 px /
`C−1` defaults and sweeps crop size (32/48/64 px) for the raw variant; rows rank by
strict-LOO rank-1, then margin. Reproduce:
`cargo run --release --bin bench_fisherface -- out/lbph/crops`
(the generated report is committed at `docs/bench-results-fisherface.md`).

## 4. Measured results

### 4.1 Hard gallery — 77 crops, 21 identities

Canonical variants, strict LOO:

| variant | margin¹ | best-threshold pair acc | EER threshold | LOO rank-1 |
|---|--:|--:|--:|--:|
| **raw (crate default)** | **8.11** | **96.5 %** @ 3.06 | 6.19 | **59/68 = 86.8 %** |
| histogram-equalised | 9.19 | 97.9 % @ 6.91 | 9.56 | 57/68 = 83.8 % |

¹ margin = mean different-identity distance − mean same-identity distance.

Distance distributions for the shipped raw variant:

| pair type | n | mean | p5 | p50 | p95 | extreme |
|---|--:|--:|--:|--:|--:|--:|
| same identity | 195 | 3.32 | 0.47 | 2.95 | 7.83 | max 14.41 |
| different identity | 2 731 | 11.44 | 4.80 | 11.42 | 17.84 | min 2.32 |

| operating point | threshold | pair accuracy | FAR | FRR |
|---|--:|--:|--:|--:|
| **crate default** `DEFAULT_MAX_DISTANCE` | **3.0** | **96.4 %** | **0.40 %** | **48.2 %** |
| best pair accuracy | 3.06 | 96.5 % | 0.40 % | 46.7 % |
| equal-error (EER) | 6.19 | — | 12.8 % | 12.8 % |

As with the other recognisers the distributions overlap (closest impostor 2.32 is
inside the genuine range), so the shipped constant is a **conservative low-FAR point**,
not an EER point: under half a percent of impostors admitted, about half of genuine
probes rejected. For the close-set question use the threshold-free
`FisherfaceRecognizer::rank` / `rank_crop` ranking — strict-LOO rank-1 is the honest
headline at **86.8 %**.

Where Fisherfaces genuinely wins is the **equal-error trade-off**: EER ≈ 12.8 %
against 14.0 % for eigenfaces and 22.5 % for LBPH on the same pairs. For a
verification deployment willing to retrain on its own gallery and re-calibrate near
the EER, it is the best of the three zero-dep options.

Calibration notes, stated plainly:

* Histogram equalisation grows the margin (8.11 → 9.19), lifts best-threshold pair
  accuracy (96.5 % → 97.9 %) and nearly halves EER (12.8 % → 6.9 %), **but costs two
  rank-1 probes** (59 → 57). Identification is the primary headline, so raw remains
  the default; `equalize: true` is one field away for verification-focused galleries.
* The one-probe gain over the PCA ceiling (59 vs 58) is within sampling noise even
  though it is the best row; it is reported as "LDA reaches and marginally exceeds the
  PCA ceiling", not as a settled accuracy improvement.

### 4.2 Crop-size sweep

| crop size | LOO rank-1 (raw) | margin | EER threshold |
|---|--:|--:|--:|
| 32 px | 58/68 = 85.3 % | 3.95 | 3.05 |
| 48 px | 58/68 = 85.3 % | 5.98 | 4.55 |
| **64 px (shipped)** | **59/68 = 86.8 %** | **8.11** | 6.19 |

64 px wins rank-1 outright and by a wide margin on separation; there is no benefit in
shrinking the crop. The component cap (`max_components`) was not swept: ranks below
`C − 1` discard exactly the least-discriminative axes of a 20-dimensional descriptor
that is already tiny.

### 4.3 Easy gallery (no-regression set) — 35 crops, 8 identities

Strict LOO with the shipped variant:

* rank-1 **33/33 = 100 %** (two singleton probes counted separately),
* distributions are fully separable: genuine max 3.23 < impostor min 3.59, so the
  interval [3.3, 3.5] gives 100 % pair accuracy with FAR = FRR = 0,
* the crate-wide constant **3.0** sits just below that gap: FAR 0 % (0/510),
  FRR 2.4 % (2/85 rejected) — a stricter point than this easy set needs, and the same
  direction (reject a few genuines, admit no impostors) the verification default fails
  on the hard set.

### 4.4 The three zero-dep recognisers on one hard-gallery line

| recogniser | descriptor | hard LOO rank-1 | hard pair EER | easy LOO rank-1 | enrolment |
|---|--:|--:|--:|--:|---|
| **LBPH** (6×6, 120 px) | 2 124 f32 | **62/68 = 91.2 %** | 22.5 % | 33/33 | incremental |
| **Fisherfaces** (64 px) | ≤ 20 f32 | 59/68 = 86.8 % | **12.8 %** | 33/33 | retrain |
| **Eigenfaces** (64 px, 98 %) | ≤ 255 f32 | 58/68 = 85.3 % | 14.0 % | 33/33 | retrain |

## 5. Honest limitations

* **Global, gallery-specific model.** A single occluding bar or a cast shadow moves
  *every* coefficient; the discriminant space itself is computed from the enrolled
  crops. The model **must be retrained when identities change** — there is deliberately
  no incremental `enroll`; call `FisherfaceRecognizer::train` again with the new crop
  list. Training is two `O(p³)` Jacobi decompositions with `p = n − C ≪ d`, cheap for
  small galleries, not for thousands of identities.
* **Needs repeated identities.** Every-singleton galleries (`n = C`) leave no
  within-class subspace; `train` returns `FisherfaceError::DegenerateGallery` rather
  than inventing a metric. Singleton classes alongside repeated ones are tolerated and
  contribute between-class signal only.
* **Threshold scale is not portable.** The coefficient L2 scale depends on crop size,
  gallery composition and the pseudo-inverse regularisation floor.
  `DEFAULT_MAX_DISTANCE = 3.0` is calibrated on the repo's two drama galleries
  (conservative low-FAR on hard, near-perfect on easy); rerun `bench_fisherface` on
  your own data before trusting it.
* **Same detection bottleneck as the other classical recognisers.** The numbers answer
  "how good are Fisherfaces given a correct box". The built-in Haar cascade is a demo
  and is not selective enough for production; supply a trained `.rfcf` cascade or use
  SCRFD. See [`recognition-lbph.md` §5](recognition-lbph.md).
* **Small, clustered dataset and no landmark alignment.** 77 crops / 21 unevenly
  populated identities from one content type; eyes and mouth are not normalised to
  fixed coordinates. Treat 86.8 % LOO rank-1 as "works on this harder domain", not an
  LFW claim. The one-probe margin over PCA does not establish LDA as generally better
  here; the robust result is the EER improvement and the tiny descriptor.
  For uncontrolled scenes the `ort-backend`/`tract-backend` ArcFace path remains the
  production recogniser.

## 6. API sketch

```rust
use rsface::fisherface::{FisherfaceConfig, FisherMatch, FisherfaceRecognizer};

// One-shot training: PCA + LDA are learned from the whole gallery. At least one
// identity must appear more than once (n − C > 0).
let samples = [
    ("alice", &alice_crop_1),
    ("alice", &alice_crop_2),
    ("bob",   &bob_crop_1),
    ("bob",   &bob_crop_2),
];
let rec = FisherfaceRecognizer::train(FisherfaceConfig::default(), samples)?;
//       face_size 64, all C−1 components, Euclidean, max_distance 3.0
assert!(rec.component_count() <= samples.len() /* ≤ C−1 */);

match rec.identify_crop(&probe) {
    FisherMatch::Match { label, distance, .. } => println!("{label} (d={distance:.2})"),
    FisherMatch::BelowThreshold { best } | FisherMatch::Ambiguous { .. }
    | FisherMatch::NoCandidates => println!("unknown"),
}
rec.verify("alice", &probe); // Option<f32>: distance if "alice" is enrolled

// Threshold-free close-set identification (the recommended primary API):
let ranking: Vec<(String, f32)> = rec.rank_crop(&probe);

// Reuse one projection for several queries:
let c = rec.project(&probe);
let ranking = rec.rank(&c);
```

Adding an identity means retraining: rebuild the sample list and call `train` again.
All inputs are `GrayImage`; use `RgbImage::to_gray()` for colour detector output.
