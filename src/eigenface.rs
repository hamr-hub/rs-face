//! Eigenfaces — the PCA recogniser (Turk & Pentland 1991), zero dependencies.
//!
//! LBPH ([`crate::lbph`]) is a hand-engineered local descriptor; eigenfaces is the
//! complementary classical approach: learn *what this gallery's faces vary along* and
//! describe every crop by its coordinates in that learned subspace. Like LBPH it needs
//! no weights download and no third-party crate — "training" is one small symmetric
//! eigendecomposition done at enrolment time.
//!
//! # Algorithm
//!
//! 1. Every crop is resized to a square (`face_size`, 64 px default) and vectorised to
//!    `d = face_size²` components after an optional histogram equalisation.
//! 2. With `n` enrolled crops, the per-pixel mean is subtracted. The covariance would
//!    be a `d × d` matrix (4096 × 4096 at the default size). Instead the `n × n` Gram
//!    matrix `G = X Xᵀ` is eigendecomposed (Jacobi rotations) and the eigenfaces are
//!    recovered as `uᵢ = Xᵀvᵢ / √λᵢ` — the standard Gram trick, exact up to rounding.
//! 3. Components are kept in descending-eigenvalue order until [`EigenfaceConfig::variance_kept`]
//!    of the energy is covered (capped at `max_components`). A crop is described by its
//!    projection coefficients `cᵢ = uᵢ·(x − mean)`.
//! 4. Matching is nearest-neighbour over enrolled crops (minimum per identity, as in
//!    [`crate::lbph::LbphRecognizer`]), with two metrics:
//!    * [`EigenMetric::Euclidean`] — plain L2 over coefficients (Turk–Pentland);
//!    * [`EigenMetric::Mahalanobis`] — coefficients divided by √λ, so low-energy
//!      detail directions count as much as the dominant (often lighting-driven) ones.
//!
//! # Accuracy envelope — be honest about this
//!
//! PCA is global: a single occluding subtitle bar, a strong cast shadow, or a pose the
//! gallery never saw moves *every* coefficient. It is also gallery-specific — the model
//! must be retrained when identities are added (there is no cheap incremental enrol the
//! way LBPH has). Treat it as the second zero-dep baseline; measured numbers for this
//! implementation are in `docs/recognition-eigenface.md`. For uncontrolled scenes use
//! the `ort-backend`/`tract-backend` ArcFace path.
//!
//! # Reference
//!
//! Turk & Pentland — *Eigenfaces for Recognition* (Journal of Cognitive Neuroscience,
//! 1991). OpenCV's `face::EigenFaceRecognizer` implements the same projection.

use std::collections::HashMap;

use crate::image::GrayImage;
use crate::linalg::{jacobi_symmetric, EIGEN_TOL_REL};

/// Default normalised crop edge in pixels (64 → 4,096-d vectors).
pub const DEFAULT_FACE_SIZE: usize = 64;

/// Cap on retained principal components regardless of gallery size. The meaningful
/// maximum is `n - 1`; this only bounds pathological galleries.
pub const DEFAULT_MAX_COMPONENTS: usize = 255;

/// Default fraction of eigenvalue energy kept by component truncation.
pub const DEFAULT_VARIANCE_KEPT: f32 = 0.98;

/// Default accept distance for the default config (64×64 crops, Euclidean metric).
///
/// A conservative **low-FAR** point checked on both of this repo's real-face
/// evaluation sets (strict per-probe LOO; see `docs/recognition-eigenface.md`):
///
/// * 35 crops / 8 identities (85 same / 510 different pairs): FAR 2.9 %, FRR 3.5 %
///   at 6.3 — it sits at the EER there (≈ 6.33, closest impostor 3.90 vs farthest
///   genuine 7.40, so no useful zero-FAR threshold exists on that set);
/// * 77 crops / 21 identities (195 same / 2 731 different pairs, harder pose and
///   lighting): FAR 1.5 %, FRR 51.8 % at 6.3; the EER is ≈ 14.0 (FAR ≈ FRR ≈ 20 %).
///
/// On the harder gallery the distributions overlap heavily, so this constant keeps
/// false accepts expensive (≤ 3 % FAR on both sets) at a high reject rate; probes
/// known to be enrolled are better handled with `rank`/`rank_crop`, whose LOO rank-1
/// is 85 % there. A 64-px/95 %-energy sweep point ties that rank-1 within one probe
/// without changing it, so the 98 % defaults are kept. The scale is gallery- and
/// crop-size-dependent — recalibrate with `bench_eigenface` per deployment.
pub const DEFAULT_MAX_DISTANCE: f32 = 6.3;

/// Distance metric used in the projection space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EigenMetric {
    /// Plain L2 distance between coefficient vectors (classic eigenfaces).
    #[default]
    Euclidean,
    /// Coefficients whitened by 1/√λ before L2 (distance-in-face-space weighting).
    Mahalanobis,
}

/// Eigenfaces training and matching parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EigenfaceConfig {
    /// Crops are resized to this square before vectorising.
    pub face_size: usize,
    /// Keep components until this fraction of eigenvalue energy is covered.
    pub variance_kept: f32,
    /// Hard cap on the number of retained components.
    pub max_components: usize,
    /// Maximum accepted match distance, in the units of [`EigenfaceConfig::metric`].
    ///
    /// [`DEFAULT_MAX_DISTANCE`] is calibrated for [`EigenMetric::Euclidean`];
    /// whitened Mahalanobis distances live on a completely different scale, so
    /// a Euclidean number cannot be reused. Prefer
    /// [`EigenfaceConfig::with_metric`] (it switches [`EigenfaceConfig::auto_threshold`]
    /// on for Mahalanobis) or recalibrate per deployment.
    pub max_distance: f32,
    /// Derive the accept threshold from the training gallery instead of
    /// [`EigenfaceConfig::max_distance`]. The derived value is the nearest
    /// between-class (impostor) coefficient distance observed at train time —
    /// i.e. the largest threshold with zero FAR on the gallery itself — which
    /// is metric-independent in the sense that it is measured in whichever
    /// distance [`EigenfaceConfig::metric`] produces. Inspect it via
    /// [`EigenfaceRecognizer::suggested_threshold`]. When the gallery contains
    /// a single class, `max_distance` is used as fallback.
    pub auto_threshold: bool,
    /// Require the best identity to beat the runner-up by at least this distance.
    pub min_margin: f32,
    /// Apply histogram equalisation before vectorising.
    pub equalize: bool,
    /// Distance metric in coefficient space.
    pub metric: EigenMetric,
}

impl EigenfaceConfig {
    /// Build a default config for the given metric. The Euclidean config is
    /// identical to [`Default::default`]; the Mahalanobis config turns
    /// [`EigenfaceConfig::auto_threshold`] on because the calibrated
    /// [`DEFAULT_MAX_DISTANCE`] (6.3) only has meaning for Euclidean distances.
    #[must_use]
    pub fn with_metric(metric: EigenMetric) -> Self {
        Self {
            metric,
            auto_threshold: metric == EigenMetric::Mahalanobis,
            ..Self::default()
        }
    }
}

impl Default for EigenfaceConfig {
    fn default() -> Self {
        Self {
            face_size: DEFAULT_FACE_SIZE,
            variance_kept: DEFAULT_VARIANCE_KEPT,
            max_components: DEFAULT_MAX_COMPONENTS,
            max_distance: DEFAULT_MAX_DISTANCE,
            auto_threshold: false,
            min_margin: 0.0,
            equalize: false,
            metric: EigenMetric::default(),
        }
    }
}

/// Reasons [`EigenfaceRecognizer::train`] refuses a gallery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EigenfaceError {
    /// Fewer than two crops were supplied — no within/across-identity axis exists.
    TooFewSamples(usize),
    /// Every crop vectorises identically (e.g. all-constant images); the Gram matrix
    /// has no usable positive eigenvalue, or the Jacobi eigensolver exhausted its
    /// sweep budget without reaching tolerance.
    DegenerateGallery,
}

impl std::fmt::Display for EigenfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EigenfaceError::TooFewSamples(n) => {
                write!(f, "eigenfaces needs >= 2 crops, got {n}")
            }
            EigenfaceError::DegenerateGallery => {
                write!(
                    f,
                    "degenerate gallery: crops vectorise identically or eigensolver did not converge"
                )
            }
        }
    }
}

impl std::error::Error for EigenfaceError {}

/// One labelled training crop.
pub type EigenSample<'a> = (&'a str, &'a GrayImage);

/// Outcome of an eigenfaces gallery query.
///
/// All zero-dep recognisers share [`crate::recognizer::Recognition`]; this
/// alias keeps the per-recogniser name working and lets eigenfaces
/// dispatch through [`crate::recognizer::FaceRecognizer`].
pub use crate::recognizer::Recognition as EigenMatch;

/// One retained principal component, stored in pixel space for probe projection.
#[derive(Clone, Debug)]
pub(crate) struct Component {
    /// Unit eigenface in the `d`-dimensional pixel space.
    pub(crate) axis: Vec<f32>,
    /// Whitening factor `1 / √λ_cov` for [`EigenMetric::Mahalanobis`].
    pub(crate) inv_scale: f32,
}

/// One enrolled crop with its projection coefficients.
#[derive(Clone, Debug)]
pub(crate) struct Member {
    pub(crate) label: String,
    pub(crate) coeffs: Vec<f32>,
}

/// Trained PCA recogniser.
///
/// Adding or removing identities changes the subspace, so this is deliberately a
/// train-once value: call [`EigenfaceRecognizer::train`] again with the new crop list.
/// The trained model can be persisted without the original crops via
/// [`EigenfaceRecognizer::to_bytes`] / [`EigenfaceRecognizer::save`] (format documented
/// in `docs/gallery-persistence.md`).
#[derive(Clone, Debug)]
pub struct EigenfaceRecognizer {
    config: EigenfaceConfig,
    pub(crate) mean: Vec<f32>,
    pub(crate) components: Vec<Component>,
    pub(crate) members: Vec<Member>,
    by_label: HashMap<String, usize>,
    /// Nearest between-class training distance (zero-training-FAR point),
    /// used when [`EigenfaceConfig::auto_threshold`] is set. `None` for a
    /// single-class gallery.
    pub(crate) suggested_threshold: Option<f32>,
}

impl EigenfaceRecognizer {
    /// Train on labelled crops. All images are resized/equalised per `config`.
    pub fn train<'a, I>(config: EigenfaceConfig, samples: I) -> Result<Self, EigenfaceError>
    where
        I: IntoIterator<Item = EigenSample<'a>>,
    {
        let crops: Vec<(&str, &GrayImage)> = samples.into_iter().collect();
        let n = crops.len();
        if n < 2 {
            return Err(EigenfaceError::TooFewSamples(n));
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

        // Gram matrix G = X X^T (n x n), then symmetric Jacobi eigendecomposition.
        let mut gram = vec![0.0f32; n * n];
        for i in 0..n {
            for j in i..n {
                let g: f32 = rows[i].iter().zip(&rows[j]).map(|(a, b)| a * b).sum();
                gram[i * n + j] = g;
                gram[j * n + i] = g;
            }
        }
        let (eigvals, eigvecs, converged) = jacobi_symmetric(&mut gram, n);
        if !converged {
            return Err(EigenfaceError::DegenerateGallery);
        }

        // Positive eigenvalues, strongest first.
        let lambda_max = eigvals.iter().cloned().fold(0.0f32, f32::max);
        let positive: Vec<(f32, Vec<f32>)> = eigvals
            .into_iter()
            .zip(eigvecs_columns(eigvecs, n))
            .filter(|(lam, _)| *lam > lambda_max.max(1.0) * EIGEN_TOL_REL)
            .collect();
        if positive.is_empty() {
            return Err(EigenfaceError::DegenerateGallery);
        }
        let total_energy: f32 = positive.iter().map(|(lam, _)| lam).sum();

        // Energy-based truncation with a hard component cap.
        let keep = select_component_count(&positive, total_energy, &config);
        let mut components = Vec::with_capacity(keep);
        let denom = (n - 1) as f32; // covariance eigenvalues are lambda_G / (n - 1)
        for (lambda_g, v) in positive.into_iter().take(keep) {
            // u = X^T v / sqrt(lambda_G); unit-length because v^T G v = lambda_G.
            let mut axis = vec![0.0f32; dim];
            for (i, row) in rows.iter().enumerate() {
                let vi = v[i];
                if vi == 0.0 {
                    continue;
                }
                for (u, &x) in axis.iter_mut().zip(row) {
                    *u += vi * x;
                }
            }
            let norm = lambda_g.sqrt();
            if norm > 0.0 {
                for u in axis.iter_mut() {
                    *u /= norm;
                }
            }
            let inv_scale = (denom / lambda_g).sqrt();
            components.push(Component { axis, inv_scale });
        }

        // Project every training crop once.
        let mut by_label = HashMap::with_capacity(n);
        let mut members = Vec::with_capacity(n);
        for (i, (label, _)) in crops.iter().enumerate() {
            let coeffs = project_centered(&rows[i], &components);
            by_label.entry(label.to_string()).or_insert(members.len());
            members.push(Member {
                label: (*label).to_string(),
                coeffs,
            });
        }

        // Nearest between-class coefficient distance: the largest threshold
        // that mis-accepts no training impostor. Computed in the configured
        // metric, so it is valid for both Euclidean and Mahalanobis scales.
        let mut suggested_threshold: Option<f32> = None;
        for a in 0..n {
            for b in (a + 1)..n {
                if members[a].label != members[b].label {
                    let d = coeff_metric_distance(
                        &members[a].coeffs,
                        &members[b].coeffs,
                        &components,
                        config.metric,
                    );
                    suggested_threshold =
                        Some(suggested_threshold.map_or(d, |cur: f32| cur.min(d)));
                }
            }
        }

        Ok(Self {
            config,
            mean,
            components,
            members,
            by_label,
            suggested_threshold,
        })
    }

    /// Rebuild from persisted trained-model parts; the codec in
    /// [`crate::subspace_store`] has already validated every shape. The label
    /// index is rebuilt to first-occurrence member indices exactly as `train`
    /// builds it from the input order.
    pub(crate) fn from_parts(
        config: EigenfaceConfig,
        mean: Vec<f32>,
        components: Vec<Component>,
        members: Vec<Member>,
        suggested_threshold: Option<f32>,
    ) -> Self {
        let mut by_label = HashMap::with_capacity(members.len());
        for (i, m) in members.iter().enumerate() {
            by_label.entry(m.label.clone()).or_insert(i);
        }
        Self {
            config,
            mean,
            components,
            members,
            by_label,
            suggested_threshold,
        }
    }

    /// Serialize the trained model (mean, eigenfaces, projected gallery,
    /// config) to the compact binary format documented in
    /// `docs/gallery-persistence.md`; the original crops are not needed.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        crate::subspace_store::encode_eigen(self)
    }

    /// Rebuild a recogniser from [`EigenfaceRecognizer::to_bytes`] output,
    /// validating the whole blob. Distances after a reload are bit-identical.
    ///
    /// # Errors
    ///
    /// [`crate::subspace_store::SubspaceStoreError`] for a truncated, corrupt,
    /// wrong-version, or internally inconsistent blob.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, crate::subspace_store::SubspaceStoreError> {
        crate::subspace_store::decode_eigen(bytes)
    }

    /// Atomically write the trained model (temp file then rename).
    ///
    /// # Errors
    ///
    /// [`crate::subspace_store::SubspaceStoreError::Io`] on filesystem failure.
    pub fn save(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), crate::subspace_store::SubspaceStoreError> {
        crate::subspace_store::save(&self.to_bytes(), path)
    }

    /// Load a trained model written by [`EigenfaceRecognizer::save`].
    ///
    /// # Errors
    ///
    /// Filesystem errors or any [`crate::subspace_store::SubspaceStoreError`].
    pub fn load(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::subspace_store::SubspaceStoreError> {
        crate::subspace_store::load(path, crate::subspace_store::decode_eigen)
    }

    /// The gallery-derived zero-training-FAR threshold (nearest between-class
    /// coefficient distance), measured in the configured metric. Used
    /// automatically when [`EigenfaceConfig::auto_threshold`] is set; callers
    /// can also read it to calibrate an explicit `max_distance`. `None` when
    /// the gallery contains a single identity.
    pub fn suggested_threshold(&self) -> Option<f32> {
        self.suggested_threshold
    }

    /// Threshold actually applied to an identification decision.
    fn accept_threshold(&self) -> f32 {
        if self.config.auto_threshold {
            self.suggested_threshold.unwrap_or(self.config.max_distance)
        } else {
            self.config.max_distance
        }
    }

    pub fn config(&self) -> &EigenfaceConfig {
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

    /// Number of retained principal components.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Project a crop into the eigenface coefficient space.
    pub fn project(&self, crop: &GrayImage) -> Vec<f32> {
        let mut v = vectorize(crop, &self.config);
        for (x, m) in v.iter_mut().zip(&self.mean) {
            *x -= *m;
        }
        project_centered(&v, &self.components)
    }

    /// Distance of `crop` to every identity (best member per identity), nearest first.
    pub fn rank_crop(&self, crop: &GrayImage) -> Vec<(String, f32)> {
        self.rank(&self.project(crop))
    }

    /// Rank an already-projected coefficient vector. A vector of the wrong length
    /// (e.g. an empty probe) yields no candidates rather than a bogus zero distance.
    pub fn rank(&self, probe: &[f32]) -> Vec<(String, f32)> {
        if probe.len() != self.components.len() {
            return Vec::new();
        }
        let mut best: HashMap<&str, f32> = HashMap::new();
        for member in &self.members {
            let d = self.coeff_distance(probe, &member.coeffs);
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
    pub fn identify_crop(&self, crop: &GrayImage) -> EigenMatch {
        self.identify(&self.project(crop))
    }

    /// Identify an already-projected coefficient vector.
    pub fn identify(&self, probe: &[f32]) -> EigenMatch {
        let ranked = self.rank(probe);
        let Some((best_label, best_dist)) = ranked.first().cloned() else {
            return EigenMatch::NoCandidates;
        };
        if best_dist > self.accept_threshold() {
            return EigenMatch::BelowThreshold {
                best: Some((best_label, best_dist)),
            };
        }
        let (margin, runner_up) = match ranked.get(1) {
            Some((l, d)) => (d - best_dist, Some(l.clone())),
            None => (f32::INFINITY, None),
        };
        if self.config.min_margin > 0.0 && margin < self.config.min_margin {
            return EigenMatch::Ambiguous {
                first: best_label,
                second: runner_up.unwrap_or_default(),
                margin,
            };
        }
        EigenMatch::Match {
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
            .map(|m| self.coeff_distance(&probe, &m.coeffs))
            .fold(None, |acc: Option<f32>, d| {
                Some(acc.map_or(d, |a| a.min(d)))
            })
    }

    /// Distance between two projected coefficient vectors under the configured metric.
    /// Both vectors must come from [`EigenfaceRecognizer::project`] on THIS model
    /// (debug builds assert the length).
    pub fn coefficient_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), self.components.len());
        debug_assert_eq!(b.len(), self.components.len());
        self.coeff_distance(a, b)
    }

    fn coeff_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        coeff_metric_distance(a, b, &self.components, self.config.metric)
    }
}

/// L2 between coefficient vectors, optionally whitened by each component's
/// `1/√λ` ([`EigenMetric::Mahalanobis`]). Shared by matching and by the
/// train-time threshold derivation.
fn coeff_metric_distance(
    a: &[f32],
    b: &[f32],
    components: &[Component],
    metric: EigenMetric,
) -> f32 {
    let sum: f32 = a
        .iter()
        .zip(b)
        .enumerate()
        .map(|(i, (x, y))| {
            let mut d = x - y;
            if metric == EigenMetric::Mahalanobis {
                d *= components[i].inv_scale;
            }
            d * d
        })
        .sum();
    sum.sqrt()
}

// ---------------------------------------------------------------------------
// Vectorisation / projection
// ---------------------------------------------------------------------------

/// Resize (and optionally equalise) a crop, then flatten pixels to `[0, 1]` f32.
fn vectorize(img: &GrayImage, config: &EigenfaceConfig) -> Vec<f32> {
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

fn project_centered(centered: &[f32], components: &[Component]) -> Vec<f32> {
    components
        .iter()
        .map(|c| c.axis.iter().zip(centered).map(|(u, x)| u * x).sum())
        .collect()
}

/// Choose how many leading components to keep: cover `variance_kept` of the energy,
/// capped by `max_components`, always keeping at least one.
fn select_component_count(
    positive: &[(f32, Vec<f32>)],
    total_energy: f32,
    config: &EigenfaceConfig,
) -> usize {
    let target = total_energy * config.variance_kept.clamp(0.0, 1.0);
    let mut cum = 0.0;
    let mut keep = 1;
    for (i, (lam, _)) in positive.iter().enumerate() {
        cum += lam;
        keep = i + 1;
        if cum >= target || keep >= config.max_components {
            break;
        }
    }
    keep.min(positive.len())
}

/// Columns of a row-major square matrix, returned as owned vectors.
fn eigvecs_columns(flat: Vec<f32>, n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|col| (0..n).map(|row| flat[row * n + col]).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_rejects_tiny_or_degenerate_galleries() {
        let cfg = EigenfaceConfig {
            face_size: 16,
            ..EigenfaceConfig::default()
        };
        let img = GrayImage::new(16, 16);
        assert_eq!(
            EigenfaceRecognizer::train(cfg, vec![("a", &img)]).unwrap_err(),
            EigenfaceError::TooFewSamples(1)
        );
        let same = GrayImage::new(16, 16);
        assert_eq!(
            EigenfaceRecognizer::train(cfg, vec![("a", &img), ("b", &same)]).unwrap_err(),
            EigenfaceError::DegenerateGallery
        );
    }

    #[test]
    fn trains_and_identifies_two_separable_classes() {
        let cfg = EigenfaceConfig {
            face_size: 24,
            variance_kept: 0.999,
            ..EigenfaceConfig::default()
        };
        // Class A: left half bright. Class B: top half bright. Within-class copies
        // differ by a small deterministic offset the PCA space must still span.
        let a1 = half_lit(24, true, 0);
        let a2 = half_lit(24, true, 3);
        let b1 = half_lit(24, false, 0);
        let b2 = half_lit(24, false, 3);
        let rec =
            EigenfaceRecognizer::train(cfg, vec![("a", &a1), ("a", &a2), ("b", &b1), ("b", &b2)])
                .expect("train");
        assert!(rec.component_count() >= 1);
        assert_eq!(rec.len(), 2);
        assert_eq!(rec.crop_count(), 4);

        // A training member must verify against its own label with (near-)zero gap...
        // truncation leaves a residual, so check ordering rather than exact zero.
        let da = rec.verify("a", &a1).unwrap();
        let db = rec.verify("b", &a1).unwrap();
        assert!(da < db, "own class {da} should beat other class {db}");
        match rec.identify_crop(&a2) {
            EigenMatch::Match { label, .. } => assert_eq!(label, "a"),
            other => panic!("expected Match(a), got {other:?}"),
        }
        match rec.identify_crop(&b1) {
            EigenMatch::Match { label, .. } => assert_eq!(label, "b"),
            other => panic!("expected Match(b), got {other:?}"),
        }
        assert!(rec.verify("ghost", &a1).is_none());
    }

    #[test]
    fn mahalanobis_and_euclidean_agree_on_ranking_for_separable_case() {
        let cfg = EigenfaceConfig {
            face_size: 24,
            variance_kept: 1.0,
            ..EigenfaceConfig::default()
        };
        let mut mah_cfg = cfg;
        mah_cfg.metric = EigenMetric::Mahalanobis;

        let crops = [
            ("a", half_lit(24, true, 0)),
            ("a", half_lit(24, true, 2)),
            ("b", half_lit(24, false, 0)),
            ("b", half_lit(24, false, 2)),
        ];
        let refs: Vec<(&str, &GrayImage)> = crops.iter().map(|(l, i)| (*l, i)).collect();
        let probe = half_lit(24, true, 5);
        let eu = EigenfaceRecognizer::train(cfg, refs.clone()).unwrap();
        let mah = EigenfaceRecognizer::train(mah_cfg, refs).unwrap();
        assert_eq!(eu.rank_crop(&probe)[0].0, "a");
        assert_eq!(mah.rank_crop(&probe)[0].0, "a");
    }

    #[test]
    fn with_metric_enables_auto_threshold_only_for_mahalanobis() {
        let eu = EigenfaceConfig::with_metric(EigenMetric::Euclidean);
        assert_eq!(eu.metric, EigenMetric::Euclidean);
        assert!(!eu.auto_threshold);
        let mah = EigenfaceConfig::with_metric(EigenMetric::Mahalanobis);
        assert_eq!(mah.metric, EigenMetric::Mahalanobis);
        assert!(mah.auto_threshold);
    }

    #[test]
    fn suggested_threshold_is_nearest_impostor_distance() {
        let cfg = EigenfaceConfig {
            face_size: 24,
            variance_kept: 1.0,
            ..EigenfaceConfig::default()
        };
        let imgs = [
            half_lit(24, true, 0),
            half_lit(24, true, 2),
            half_lit(24, false, 0),
            half_lit(24, false, 2),
        ];
        let crops: Vec<(&str, &GrayImage)> = vec![
            ("a", &imgs[0]),
            ("a", &imgs[1]),
            ("b", &imgs[2]),
            ("b", &imgs[3]),
        ];
        let rec = EigenfaceRecognizer::train(cfg, crops).unwrap();
        let suggested = rec.suggested_threshold().expect("multi-class gallery");
        assert!(suggested.is_finite() && suggested > 0.0);
        // Every cross-class pair must sit at or above the derived threshold,
        // which is defined as their minimum (`distance > thr` rejects).
        let a_proj = rec.project(&half_lit(24, true, 0));
        let b_proj = rec.project(&half_lit(24, false, 0));
        let d = rec.coefficient_distance(&a_proj, &b_proj);
        assert!(
            d >= suggested - 1e-4,
            "cross-class distance {d} must not be below the threshold {suggested}"
        );
    }

    #[test]
    fn mahalanobis_auto_threshold_identifies_separable_gallery() {
        // Regression: Mahalanobis distances are on a different scale than
        // Euclidean, so the calibrated 6.3 default would be meaningless; the
        // auto threshold must follow the whitened scale and keep a separable
        // gallery identifiable.
        let cfg = EigenfaceConfig::with_metric(EigenMetric::Mahalanobis);
        let imgs = [
            half_lit(24, true, 0),
            half_lit(24, true, 2),
            half_lit(24, false, 0),
            half_lit(24, false, 2),
        ];
        let crops: Vec<(&str, &GrayImage)> = vec![
            ("a", &imgs[0]),
            ("a", &imgs[1]),
            ("b", &imgs[2]),
            ("b", &imgs[3]),
        ];
        let rec = EigenfaceRecognizer::train(
            EigenfaceConfig {
                face_size: 24,
                variance_kept: 1.0,
                ..cfg
            },
            crops,
        )
        .unwrap();
        assert!(rec.suggested_threshold().is_some());
        match rec.identify_crop(&half_lit(24, true, 1)) {
            EigenMatch::Match { label, .. } => assert_eq!(label, "a"),
            other => panic!("expected Match(a) under Mahalanobis auto threshold, got {other:?}"),
        }
    }

    #[test]
    fn single_class_gallery_has_no_suggested_threshold_and_falls_back() {
        let cfg = EigenfaceConfig {
            face_size: 24,
            auto_threshold: true,
            max_distance: 50.0,
            variance_kept: 1.0,
            ..EigenfaceConfig::default()
        };
        let rec = EigenfaceRecognizer::train(
            cfg,
            vec![("a", &half_lit(24, true, 0)), ("a", &half_lit(24, true, 2))],
        )
        .unwrap();
        assert_eq!(rec.suggested_threshold(), None);
        // Fallback is the explicit max_distance, not a panic / zero threshold.
        match rec.identify_crop(&half_lit(24, true, 4)) {
            EigenMatch::Match { label, .. } => assert_eq!(label, "a"),
            other => panic!("expected fallback acceptance, got {other:?}"),
        }
    }

    #[test]
    fn threshold_policy_rejects_unknown_face() {
        let cfg = EigenfaceConfig {
            face_size: 24,
            max_distance: 0.0, // nothing but an exact projection can pass
            ..EigenfaceConfig::default()
        };
        let rec = EigenfaceRecognizer::train(
            cfg,
            vec![
                ("a", &half_lit(24, true, 0)),
                ("b", &half_lit(24, false, 0)),
            ],
        )
        .unwrap();
        match rec.identify_crop(&half_lit(24, true, 9)) {
            EigenMatch::BelowThreshold { .. } => {}
            // A Match can only occur if the probe projects exactly onto a member.
            EigenMatch::Match { distance, .. } => assert!(distance > 0.0),
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(rec.identify(&[]), EigenMatch::NoCandidates);
    }

    #[test]
    fn projection_dimension_matches_component_count() {
        let cfg = EigenfaceConfig {
            face_size: 16,
            max_components: 2,
            variance_kept: 1.0,
            ..EigenfaceConfig::default()
        };
        let rec = EigenfaceRecognizer::train(
            cfg,
            vec![
                ("a", &half_lit(16, true, 0)),
                ("a", &half_lit(16, true, 1)),
                ("b", &half_lit(16, false, 0)),
                ("b", &half_lit(16, false, 1)),
            ],
        )
        .unwrap();
        assert!(rec.component_count() <= 2);
        assert_eq!(
            rec.project(&half_lit(16, true, 2)).len(),
            rec.component_count()
        );
    }

    /// Deterministic class pattern: a bright half (left when `left`, else top) with a
    /// fixed brightness `lift` so copies genuinely differ but stay in the same class.
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
