//! Fisherfaces — the LDA recogniser (Belhumeur, Hespanha, Kriegman 1997), zero
//! dependencies.
//!
//! Eigenfaces ([`crate::eigenface`]) maximises *total* scatter, so the strongest
//! principal directions are often lighting and pose — nuisance variation that carries
//! no identity. Fisherfaces instead learn the directions that **separate the enrolled
//! identities from each other** while keeping each identity's own crops tightly
//! clustered: it solves Fisher's linear discriminant between-class / within-class.
//! Like LBPH and eigenfaces it needs no weights download and no third-party crate.
//!
//! # Algorithm
//!
//! 1. Every crop is resized to a square (`face_size`, 64 px default) and vectorised to
//!    `d = face_size²` components in `[0, 1]` after an optional histogram
//!    equalisation; the per-pixel gallery mean is subtracted.
//! 2. With `n` crops over `C` distinct identities, the within-class scatter matrix is
//!    singular in `d`-space (d ≫ n). The textbook reduction projects onto the leading
//!    `n − C` principal directions of the total scatter first (the Gram-matrix trick
//!    from [`crate::eigenface`], solved with the shared Jacobi routine in
//!    [`crate::linalg`]). The projected within-class scatter is generically
//!    non-singular.
//! 3. In the PCA space the scatter matrices are built directly:
//!    `S_W = Σ_k Σ_{i∈k} (z_i − μ_k)(z_i − μ_k)ᵀ`,
//!    `S_B = Σ_k n_k μ_k μ_kᵀ`.
//! 4. The generalised problem `S_B w = λ S_W w` is made symmetric by whitening:
//!    `S_W^{−1/2} S_B S_W^{−1/2}` (Jacobi again; near-zero eigenvalues of `S_W` get a
//!    pseudo-inverse weight, which regularises galleries with near-duplicate frames).
//!    At most `C − 1` discriminant directions exist; the eigenvectors are transformed
//!    back to pixel space and unit-normalised.
//! 5. A crop is described by its projection coefficients and matched with Euclidean
//!    nearest neighbour (minimum distance per identity, as in the other recognisers).
//!
//! # Accuracy envelope — be honest about this
//!
//! LDA is global and gallery-specific: the model **must be retrained when identities
//! change** (there is no incremental enrol), and a single occluding bar or unseen pose
//! moves every coefficient. It needs at least two crops per identity in expectation;
//! singleton classes are tolerated but contribute only between-class signal. Measured
//! under strict per-probe LOO retrain on the repo galleries
//! (`docs/recognition-fisherface.md`): 59/68 rank-1 (86.8 %) on the 77-crop /
//! 21-identity hard set — one probe above the eigenfaces 58/68 PCA ceiling, three
//! below LBPH's 62/68 — and 33/33 on the easy set; the pair-distance EER (12.8 %) is
//! the best of the three zero-dep recognisers. For uncontrolled scenes use the
//! `ort-backend`/`tract-backend` ArcFace path.
//!
//! # Reference
//!
//! Belhumeur, Hespanha, Kriegman — *Eigenfaces vs. Fisherfaces: Recognition Using
//! Class Specific Linear Projection* (PAMI 1997). OpenCV's
//! `face::FisherFaceRecognizer` implements the same projection.

use std::collections::HashMap;

use crate::image::GrayImage;
use crate::linalg::{jacobi_symmetric, EIGEN_TOL_REL};

/// Default normalised crop edge in pixels (64 → 4,096-d vectors).
pub const DEFAULT_FACE_SIZE: usize = 64;

/// Conservative accept distance in LDA coefficient space (raw 64 px, `C − 1`
/// components): calibrated on the 77-crop / 21-identity hard gallery as a low-FAR
/// point — FAR 0.40 % / FRR 48.2 % at 96.5 % strict-LOO pair accuracy, EER ≈ 12.8 %.
/// On the easy 35-crop / 8-identity no-regression gallery the same constant gives
/// FAR 0 % / FRR 2.4 % and 33/33 rank-1. Thresholds are deployment-dependent; close-set
/// identification should use the threshold-free [`FisherfaceRecognizer::rank_crop`].
/// See `docs/recognition-fisherface.md`.
pub const DEFAULT_MAX_DISTANCE: f32 = 3.0;

/// Fisherfaces training and matching parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FisherfaceConfig {
    /// Crops are resized to this square before vectorising.
    pub face_size: usize,
    /// Hard cap on retained discriminant components. The meaningful maximum is
    /// `identities − 1`; `0` means "keep all `C − 1`".
    pub max_components: usize,
    /// Maximum accepted match distance in LDA coefficient space.
    pub max_distance: f32,
    /// Require the best identity to beat the runner-up by at least this distance.
    pub min_margin: f32,
    /// Apply histogram equalisation before vectorising.
    pub equalize: bool,
}

impl Default for FisherfaceConfig {
    /// 64 px crops, all `C − 1` discriminant components, Euclidean nearest neighbour.
    fn default() -> Self {
        Self {
            face_size: DEFAULT_FACE_SIZE,
            max_components: 0,
            max_distance: DEFAULT_MAX_DISTANCE,
            min_margin: 0.0,
            equalize: false,
        }
    }
}

/// Reasons [`FisherfaceRecognizer::train`] refuses a gallery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FisherfaceError {
    /// Fewer than two crops were supplied.
    TooFewSamples(usize),
    /// Only one distinct label is present — LDA has no between-class axis.
    SingleClass,
    /// Every identity has exactly one crop (`n − C = 0`): there is no within-class
    /// subspace to discriminatively whiten, so train at least one identity twice.
    DegenerateGallery,
}

impl std::fmt::Display for FisherfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FisherfaceError::TooFewSamples(n) => {
                write!(f, "fisherfaces needs >= 2 crops, got {n}")
            }
            FisherfaceError::SingleClass => {
                write!(f, "fisherfaces needs crops from at least 2 identities")
            }
            FisherfaceError::DegenerateGallery => write!(
                f,
                "degenerate gallery: every identity has a single crop (need one repeat)"
            ),
        }
    }
}

impl std::error::Error for FisherfaceError {}

/// One labelled training crop.
pub type FisherSample<'a> = (&'a str, &'a GrayImage);

/// Outcome of a fisherfaces gallery query, mirroring [`crate::lbph::LbphMatch`].
#[derive(Clone, Debug, PartialEq)]
pub enum FisherMatch {
    /// Best identity is within [`FisherfaceConfig::max_distance`] (and margin policy).
    Match {
        label: String,
        distance: f32,
        margin: f32,
    },
    /// Nearest identity was farther than `max_distance`.
    BelowThreshold { best: Option<(String, f32)> },
    /// Within threshold but too close to another identity to call safely.
    Ambiguous {
        first: String,
        second: String,
        margin: f32,
    },
    /// The recogniser has no trained model.
    NoCandidates,
}

/// One enrolled crop with its projection coefficients.
#[derive(Clone, Debug)]
struct Member {
    label: String,
    coeffs: Vec<f32>,
}

/// Trained Fisher-LDA recogniser.
///
/// Adding or removing identities changes the discriminant subspace, so this is a
/// train-once value: call [`FisherfaceRecognizer::train`] again with the new crop list.
#[derive(Clone, Debug)]
pub struct FisherfaceRecognizer {
    config: FisherfaceConfig,
    mean: Vec<f32>,
    /// Unit discriminant axes in the `d`-dimensional pixel space.
    axes: Vec<Vec<f32>>,
    members: Vec<Member>,
    by_label: HashMap<String, usize>,
}

impl FisherfaceRecognizer {
    /// Train on labelled crops. All images are resized/equalised per `config`.
    pub fn train<'a, I>(config: FisherfaceConfig, samples: I) -> Result<Self, FisherfaceError>
    where
        I: IntoIterator<Item = FisherSample<'a>>,
    {
        let crops: Vec<(&str, &GrayImage)> = samples.into_iter().collect();
        let n = crops.len();
        if n < 2 {
            return Err(FisherfaceError::TooFewSamples(n));
        }

        // Stable class numbering in first-seen order.
        let mut class_of: Vec<usize> = Vec::with_capacity(n);
        let mut class_names: Vec<String> = Vec::new();
        let mut class_n: Vec<usize> = Vec::new();
        for (label, _) in &crops {
            match class_names.iter().position(|l| l == label) {
                Some(k) => {
                    class_of.push(k);
                    class_n[k] += 1;
                }
                None => {
                    class_of.push(class_names.len());
                    class_names.push((*label).to_string());
                    class_n.push(1);
                }
            }
        }
        let c = class_names.len();
        if c < 2 {
            return Err(FisherfaceError::SingleClass);
        }
        // Belhumeur reduction: project onto n−C principal directions first.
        let pca_rank = n - c;
        if pca_rank < 1 {
            return Err(FisherfaceError::DegenerateGallery);
        }

        // Vectorise to centred pixel rows in [0, 1].
        let dim = config.face_size * config.face_size;
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
        for (_, img) in &crops {
            rows.push(vectorize(img, &config));
        }
        let mut mean = vec![0.0f32; dim];
        for row in &rows {
            for (m, &v) in mean.iter_mut().zip(row) {
                *m += v;
            }
        }
        let inv_n = 1.0 / n as f32;
        for m in mean.iter_mut() {
            *m *= inv_n;
        }
        for row in rows.iter_mut() {
            for (v, m) in row.iter_mut().zip(&mean) {
                *v -= *m;
            }
        }

        let pca = pca_basis(&rows, n, dim, pca_rank)?;
        let p = pca.len();

        // Project every crop into the PCA space and gather class means.
        let mut z = vec![vec![0.0f32; p]; n];
        let mut class_mean = vec![vec![0.0f32; p]; c];
        for (i, row) in rows.iter().enumerate() {
            for (k, axis) in pca.iter().enumerate() {
                let proj: f32 = axis.iter().zip(row).map(|(u, x)| u * x).sum();
                z[i][k] = proj;
                class_mean[class_of[i]][k] += proj;
            }
        }
        for k in 0..c {
            let inv = 1.0 / class_n[k] as f32;
            for v in class_mean[k].iter_mut() {
                *v *= inv;
            }
        }

        // Within-class and between-class scatter in PCA space.
        let mut sw = vec![0.0f32; p * p];
        let mut sb = vec![0.0f32; p * p];
        for (i, zi) in z.iter().enumerate() {
            let mk = &class_mean[class_of[i]];
            rank1_add(&mut sw, p, zi, mk, 1.0);
        }
        for k in 0..c {
            rank1_add(&mut sb, p, &class_mean[k], &zeros(p), class_n[k] as f32);
        }

        // Generalised eigenproblem via symmetric whitening of S_W.
        let (sw_vals, sw_vecs) = jacobi_symmetric(&mut sw, p);
        let sw_max = sw_vals.first().copied().unwrap_or(0.0).abs();
        let floor = sw_max.max(1.0) * 1e-9;
        // W_w columns: v_k / sqrt(lambda_k); zeroed for the rank-deficient tail.
        let mut whitener = vec![0.0f32; p * p];
        for k in 0..p {
            if sw_vals[k] > floor {
                let scale = sw_vals[k].sqrt().recip();
                for row in 0..p {
                    whitener[row * p + k] = sw_vecs[row * p + k] * scale;
                }
            }
        }
        // S_B' = W_w^T S_B W_w (symmetric p x p), as two O(p^3) matrix products;
        // rank-deficient whitener columns (zeroed above) stay zero throughout.
        let mut sb_w = vec![0.0f32; p * p]; // S_B W_w
        for a in 0..p {
            for b in 0..p {
                let mut acc = 0.0;
                for k in 0..p {
                    acc += sb[a * p + k] * whitener[k * p + b];
                }
                sb_w[a * p + b] = acc;
            }
        }
        let mut sb_white = vec![0.0f32; p * p];
        for i in 0..p {
            for j in i..p {
                let mut acc = 0.0;
                for a in 0..p {
                    acc += whitener[a * p + i] * sb_w[a * p + j];
                }
                sb_white[i * p + j] = acc;
                sb_white[j * p + i] = acc;
            }
        }
        let (lda_vals, lda_vecs) = jacobi_symmetric(&mut sb_white, p);

        // Keep the leading C−1 discriminant axes (config cap applied), transform back
        // to PCA space (W_w q), then to pixel space (U_pca a), and unit-normalise.
        let cap = if config.max_components == 0 {
            c - 1
        } else {
            config.max_components.min(c - 1)
        };
        let lda_floor = lda_vals.first().copied().unwrap_or(0.0).max(1.0) * EIGEN_TOL_REL;
        let mut axes: Vec<Vec<f32>> = Vec::with_capacity(cap);
        for k in 0..p {
            if axes.len() >= cap || lda_vals[k] <= lda_floor {
                break;
            }
            // a = W_w q_k in PCA space.
            let mut a = vec![0.0f32; p];
            for (r, ar) in a.iter_mut().enumerate() {
                *ar = (0..p)
                    .map(|j| whitener[r * p + j] * lda_vecs[j * p + k])
                    .sum();
            }
            // f = U_pca a in pixel space; U_pca rows are axes[k][pixel].
            let mut axis = vec![0.0f32; dim];
            for (j, aj) in a.iter().enumerate() {
                if *aj == 0.0 {
                    continue;
                }
                for (f, u) in axis.iter_mut().zip(&pca[j]) {
                    *f += aj * u;
                }
            }
            let norm = axis.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for f in axis.iter_mut() {
                    *f /= norm;
                }
                axes.push(axis);
            }
        }
        if axes.is_empty() {
            return Err(FisherfaceError::DegenerateGallery);
        }

        // Project every training crop onto the discriminant axes.
        let mut by_label = HashMap::with_capacity(c);
        let mut members = Vec::with_capacity(n);
        for (i, (label, _)) in crops.iter().enumerate() {
            let coeffs = project_centered(&rows[i], &axes);
            by_label.entry(label.to_string()).or_insert(members.len());
            members.push(Member {
                label: (*label).to_string(),
                coeffs,
            });
        }

        Ok(Self {
            config,
            mean,
            axes,
            members,
            by_label,
        })
    }

    pub fn config(&self) -> &FisherfaceConfig {
        &self.config
    }

    /// Number of distinct enrolled identities.
    pub fn len(&self) -> usize {
        self.by_label.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_label.is_empty()
    }

    /// Total crops the model was trained on.
    pub fn crop_count(&self) -> usize {
        self.members.len()
    }

    /// Number of retained discriminant components (≤ identities − 1).
    pub fn component_count(&self) -> usize {
        self.axes.len()
    }

    /// Project a crop into the Fisherface coefficient space.
    pub fn project(&self, crop: &GrayImage) -> Vec<f32> {
        let mut v = vectorize(crop, &self.config);
        for (x, m) in v.iter_mut().zip(&self.mean) {
            *x -= *m;
        }
        project_centered(&v, &self.axes)
    }

    /// Distance of `crop` to every identity (best member per identity), nearest first.
    pub fn rank_crop(&self, crop: &GrayImage) -> Vec<(String, f32)> {
        self.rank(&self.project(crop))
    }

    /// Rank an already-projected coefficient vector. A vector of the wrong length
    /// (e.g. an empty probe) yields no candidates rather than a bogus zero distance.
    pub fn rank(&self, probe: &[f32]) -> Vec<(String, f32)> {
        if probe.len() != self.axes.len() {
            return Vec::new();
        }
        let mut best: HashMap<&str, f32> = HashMap::new();
        for member in &self.members {
            let d = coeff_distance(probe, &member.coeffs);
            best.entry(member.label.as_str())
                .and_modify(|cur| *cur = cur.min(d))
                .or_insert(d);
        }
        let mut scored: Vec<(String, f32)> = best
            .into_iter()
            .map(|(label, d)| (label.to_owned(), d))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored
    }

    /// Identify a crop under the configured distance/margin policy.
    pub fn identify_crop(&self, crop: &GrayImage) -> FisherMatch {
        self.identify(&self.project(crop))
    }

    /// Identify an already-projected coefficient vector.
    pub fn identify(&self, probe: &[f32]) -> FisherMatch {
        let ranked = self.rank(probe);
        let Some((best_label, best_dist)) = ranked.first().cloned() else {
            return FisherMatch::NoCandidates;
        };
        if best_dist > self.config.max_distance {
            return FisherMatch::BelowThreshold {
                best: Some((best_label, best_dist)),
            };
        }
        let (margin, runner_up) = match ranked.get(1) {
            Some((l, d)) => (d - best_dist, Some(l.clone())),
            None => (f32::INFINITY, None),
        };
        if self.config.min_margin > 0.0 && margin < self.config.min_margin {
            return FisherMatch::Ambiguous {
                first: best_label,
                second: runner_up.unwrap_or_default(),
                margin,
            };
        }
        FisherMatch::Match {
            label: best_label,
            distance: best_dist,
            margin,
        }
    }

    /// One-to-one verification: coefficient distance to the claimed label's nearest
    /// enrolled crop. `None` if that label was not trained.
    pub fn verify(&self, label: &str, crop: &GrayImage) -> Option<f32> {
        if !self.by_label.contains_key(label) {
            return None;
        }
        let probe = self.project(crop);
        self.members
            .iter()
            .filter(|m| m.label == label)
            .map(|m| coeff_distance(&probe, &m.coeffs))
            .fold(None, |acc: Option<f32>, d| {
                Some(acc.map_or(d, |a| a.min(d)))
            })
    }

    /// Distance between two projected coefficient vectors on THIS model (debug builds
    /// assert the length).
    pub fn coefficient_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), self.axes.len());
        debug_assert_eq!(b.len(), self.axes.len());
        coeff_distance(a, b)
    }
}

// ---------------------------------------------------------------------------
// Vectorisation / scatter helpers
// ---------------------------------------------------------------------------

/// Resize (and optionally equalise) a crop, then flatten pixels to `[0, 1]` f32.
fn vectorize(img: &GrayImage, config: &FisherfaceConfig) -> Vec<f32> {
    let need_resize = img.width() != config.face_size || img.height() != config.face_size;
    let mut face = if need_resize {
        img.resize_bilinear(config.face_size, config.face_size)
    } else {
        GrayImage::from_vec(img.as_slice().to_vec(), img.width(), img.height())
    };
    if config.equalize {
        face.equalize_hist_inplace();
    }
    face.as_slice()
        .iter()
        .map(|&p| f32::from(p) / 255.0)
        .collect()
}

fn project_centered(centered: &[f32], axes: &[Vec<f32>]) -> Vec<f32> {
    axes.iter()
        .map(|axis| axis.iter().zip(centered).map(|(u, x)| u * x).sum())
        .collect()
}

fn zeros(p: usize) -> Vec<f32> {
    vec![0.0; p]
}

/// `M += weight · (a − b)(a − b)ᵀ` on a packed `p × p` symmetric matrix.
fn rank1_add(m: &mut [f32], p: usize, a: &[f32], b: &[f32], weight: f32) {
    debug_assert_eq!(a.len(), p);
    debug_assert_eq!(b.len(), p);
    for i in 0..p {
        let ai = (a[i] - b[i]) * weight;
        for j in i..p {
            let v = ai * (a[j] - b[j]);
            m[i * p + j] += v;
            m[j * p + i] += v;
        }
    }
}

/// Leading `rank` orthonormal PCA axes of centred pixel `rows` (n rows, dim columns)
/// via the n×n Gram trick; each returned axis is a `dim`-length unit vector.
fn pca_basis(
    rows: &[Vec<f32>],
    n: usize,
    dim: usize,
    rank: usize,
) -> Result<Vec<Vec<f32>>, FisherfaceError> {
    let mut gram = vec![0.0f32; n * n];
    for i in 0..n {
        for j in i..n {
            let g: f32 = rows[i].iter().zip(&rows[j]).map(|(a, b)| a * b).sum();
            gram[i * n + j] = g;
            gram[j * n + i] = g;
        }
    }
    let (eigvals, eigvecs) = jacobi_symmetric(&mut gram, n);
    let lambda_max = eigvals.first().copied().unwrap_or(0.0).abs();
    let tol = lambda_max.max(1.0) * EIGEN_TOL_REL;

    let mut axes = Vec::with_capacity(rank);
    for k in 0..n {
        if axes.len() >= rank {
            break;
        }
        let lambda_g = eigvals[k];
        if lambda_g <= tol {
            continue;
        }
        let mut axis = vec![0.0f32; dim];
        for (i, row) in rows.iter().enumerate() {
            let vi = eigvecs[i * n + k];
            if vi == 0.0 {
                continue;
            }
            for (u, &x) in axis.iter_mut().zip(row) {
                *u += vi * x;
            }
        }
        let norm = lambda_g.sqrt();
        for u in axis.iter_mut() {
            *u /= norm;
        }
        axes.push(axis);
    }
    if axes.is_empty() {
        Err(FisherfaceError::DegenerateGallery)
    } else {
        Ok(axes)
    }
}

fn coeff_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_rejects_tiny_or_degenerate_galleries() {
        let cfg = FisherfaceConfig {
            face_size: 16,
            ..FisherfaceConfig::default()
        };
        let img = GrayImage::new(16, 16);
        assert_eq!(
            FisherfaceRecognizer::train(cfg, vec![("a", &img)]).unwrap_err(),
            FisherfaceError::TooFewSamples(1)
        );
        let same = GrayImage::new(16, 16);
        assert_eq!(
            FisherfaceRecognizer::train(cfg, vec![("a", &img), ("a", &same)]).unwrap_err(),
            FisherfaceError::SingleClass
        );
        // Two distinct singleton classes: n−C = 0, no within-class subspace.
        let other = half_lit(16, false, 0);
        assert_eq!(
            FisherfaceRecognizer::train(cfg, vec![("a", &img), ("b", &other)]).unwrap_err(),
            FisherfaceError::DegenerateGallery
        );
    }

    #[test]
    fn trains_and_identifies_two_separable_classes() {
        let cfg = FisherfaceConfig {
            face_size: 24,
            ..FisherfaceConfig::default()
        };
        let a1 = half_lit(24, true, 0);
        let a2 = half_lit(24, true, 3);
        let b1 = half_lit(24, false, 0);
        let b2 = half_lit(24, false, 3);
        let rec =
            FisherfaceRecognizer::train(cfg, vec![("a", &a1), ("a", &a2), ("b", &b1), ("b", &b2)])
                .expect("train");
        // C − 1 = 1 discriminant direction.
        assert_eq!(rec.component_count(), 1);
        assert_eq!(rec.len(), 2);
        assert_eq!(rec.crop_count(), 4);

        let da = rec.verify("a", &a1).unwrap();
        let db = rec.verify("b", &a1).unwrap();
        assert!(da < db, "own class {da} should beat other class {db}");
        match rec.identify_crop(&a2) {
            FisherMatch::Match { label, .. } => assert_eq!(label, "a"),
            other => panic!("expected Match(a), got {other:?}"),
        }
        match rec.identify_crop(&b1) {
            FisherMatch::Match { label, .. } => assert_eq!(label, "b"),
            other => panic!("expected Match(b), got {other:?}"),
        }
        assert!(rec.verify("ghost", &a1).is_none());
    }

    #[test]
    fn singleton_classes_are_tolerated() {
        // n = 5, C = 3: one class is a singleton (as in the real eval gallery). The
        // recogniser must train, keep C−1 = 2 discriminant directions, and still order
        // the repeated classes correctly. The repeated copies carry independent
        // per-pixel noise so the within-class scatter fills the n−C = 2 PCA subspace
        // (a uniform-lift toy pattern would not — it is the DC direction the PCA
        // reduction removes).
        let cfg = FisherfaceConfig {
            face_size: 24,
            ..FisherfaceConfig::default()
        };
        // Checkerboard: mean sits near the grey midpoint (so global centering does not
        // erase the class) but the spatial structure is orthogonal to the half-lit
        // classes, giving the third class its own discriminant direction.
        let mut s = GrayImage::new(24, 24);
        for y in 0..24 {
            for x in 0..24 {
                s.as_mut_slice()[y * 24 + x] = if (x + y) % 2 == 0 { 240 } else { 20 };
            }
        }
        // Independent ±50 pseudo-random noise per repeated class fills the n−C PCA
        // subspace with within-class energy, as real face texture does.
        let a1 = noisy_half(24, true, 11, 1);
        let a2 = noisy_half(24, true, 11, -1);
        let b1 = noisy_half(24, false, 29, 1);
        let b2 = noisy_half(24, false, 29, -1);
        let samples = vec![("a", &a1), ("a", &a2), ("b", &b1), ("b", &b2), ("s", &s)];
        let rec = FisherfaceRecognizer::train(cfg, samples).expect("train with singleton");
        assert_eq!(rec.component_count(), 2);
        assert_eq!(rec.len(), 3);
        assert_eq!(rec.rank_crop(&half_lit(24, true, 4))[0].0, "a");
        assert_eq!(rec.rank_crop(&half_lit(24, false, 4))[0].0, "b");
    }

    #[test]
    fn component_cap_and_projection_lengths() {
        let cfg = FisherfaceConfig {
            face_size: 16,
            max_components: 1,
            ..FisherfaceConfig::default()
        };
        let rec = FisherfaceRecognizer::train(
            cfg,
            vec![
                ("a", &half_lit(16, true, 0)),
                ("a", &half_lit(16, true, 1)),
                ("b", &half_lit(16, false, 0)),
                ("b", &half_lit(16, false, 1)),
                ("c", &half_lit(16, false, 7)),
                ("c", &half_lit(16, false, 8)),
            ],
        )
        .unwrap();
        assert_eq!(rec.component_count(), 1);
        assert_eq!(rec.project(&half_lit(16, true, 2)).len(), 1);
        assert_eq!(rec.identify(&[]), FisherMatch::NoCandidates);
    }

    /// `half_lit` plus deterministic ±50 pseudo-random per-pixel noise (sign flipped
    /// between the two copies of a class so the within-class scatter has real rank).
    fn noisy_half(size: usize, left: bool, seed: u64, sign: i32) -> GrayImage {
        let mut img = half_lit(size, left, 0);
        for y in 0..size {
            for x in 0..size {
                let h = (x as u64)
                    .wrapping_mul(73_856_093)
                    .wrapping_add(y as u64 * 19_349_663)
                    .wrapping_add(seed.wrapping_mul(83_492_791));
                let noise = (h % 101) as i32 - 50;
                let v = img.as_slice()[y * size + x] as i32 + sign * noise;
                img.as_mut_slice()[y * size + x] = v.clamp(0, 255) as u8;
            }
        }
        img
    }

    /// Deterministic class pattern: a bright half (left when `left`, else top) with a
    /// fixed brightness `lift` so copies differ but stay in the same class.
    fn half_lit(size: usize, left: bool, lift: usize) -> GrayImage {
        let mut img = GrayImage::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let bright = if left { x < size / 2 } else { y < size / 2 };
                let base = if bright { 200 } else { 40 };
                img.as_mut_slice()[y * size + x] = ((base + lift) % 256) as u8;
            }
        }
        img
    }
}
