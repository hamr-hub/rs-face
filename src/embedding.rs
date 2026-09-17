//! Face embeddings and identity matching.
//!
//! An embedding is a unit-length vector in a space where cosine distance approximates
//! identity difference. Two design points matter for correctness:
//!
//! 1. **Normalisation is enforced by the type.** [`Embedding`] can only be constructed
//!    through [`Embedding::from_raw`], which L2-normalises. Once every vector is unit
//!    length, cosine similarity reduces to a plain dot product, and — more importantly —
//!    a caller cannot accidentally compare a normalised vector against a raw logit
//!    vector and get a silently meaningless number.
//!
//! 2. **Thresholds are calibrated per model, not universal.** A cosine threshold that is
//!    correct for ArcFace R50 is wrong for a MobileFaceNet, so the threshold lives in
//!    [`MatchConfig`] next to the model that produced the embeddings rather than as a
//!    global constant.

/// Embedding dimensionality of the ArcFace-family backbones (`w600k_r50`, `w600k_mbf`).
pub const ARCFACE_EMBEDDING_DIM: usize = 512;

/// An L2-normalised face embedding.
///
/// The unit-length invariant is upheld by construction, so [`Embedding::cosine`] is a
/// bare dot product with no per-call renormalisation.
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding {
    values: Vec<f32>,
}

impl Embedding {
    /// L2-normalise a raw backbone output into an [`Embedding`].
    ///
    /// Returns `None` for an empty vector or one whose norm is ~0. A zero-norm output
    /// means the forward pass collapsed (typically an all-black or NaN crop); returning
    /// `None` keeps that from entering the gallery as a vector that matches everything.
    pub fn from_raw(raw: &[f32]) -> Option<Self> {
        if raw.is_empty() {
            return None;
        }
        let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !norm.is_finite() || norm < 1e-8 {
            return None;
        }
        Some(Self {
            values: raw.iter().map(|v| v / norm).collect(),
        })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Cosine similarity in `[-1, 1]`; higher means more likely the same identity.
    ///
    /// Returns `None` on a dimension mismatch instead of comparing a prefix, so mixing
    /// embeddings from two different models is a caught error rather than a plausible
    /// but meaningless score.
    pub fn cosine(&self, other: &Embedding) -> Option<f32> {
        if self.dim() != other.dim() {
            return None;
        }
        let dot: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();
        // Both operands are unit length, so the dot product is already the cosine.
        // Clamp only to absorb floating-point drift past the mathematical bound.
        Some(dot.clamp(-1.0, 1.0))
    }

    /// Squared Euclidean distance between two unit vectors.
    ///
    /// For normalised embeddings this is exactly `2 - 2*cos`, i.e. a monotone
    /// reparametrisation of cosine similarity. Provided because much of the face
    /// literature quotes L2 thresholds; rankings are identical either way.
    pub fn sq_euclidean(&self, other: &Embedding) -> Option<f32> {
        self.cosine(other).map(|c| 2.0 - 2.0 * c)
    }
}

/// Matching policy for the gallery.
#[derive(Clone, Debug)]
pub struct MatchConfig {
    /// Minimum cosine similarity to accept a match.
    ///
    /// The default of 0.36 is InsightFace's commonly used operating point for the
    /// `w600k_r50` ArcFace backbone, chosen near TAR@FAR=1e-4. It is intentionally
    /// conservative: in an access-control setting a false accept is far costlier than a
    /// false reject. Recalibrate on your own data before trusting it in production —
    /// see `docs/recognition.md`.
    pub threshold: f32,

    /// Require the best match to beat the runner-up by this margin.
    ///
    /// Guards the case where a probe sits almost equidistant between two enrolled
    /// identities (identical twins, or a low-quality crop). Without a margin the winner
    /// is decided by noise. `0.0` disables the check.
    pub min_margin: f32,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            threshold: 0.36,
            min_margin: 0.0,
        }
    }
}

impl MatchConfig {
    pub fn with_threshold(mut self, t: f32) -> Self {
        self.threshold = t;
        self
    }

    pub fn with_min_margin(mut self, m: f32) -> Self {
        self.min_margin = m;
        self
    }
}

/// One enrolled identity, possibly with several reference embeddings.
///
/// Multiple embeddings per identity is the point: a single frontal shot generalises
/// poorly across pose and lighting, and averaging into one centroid discards the
/// multi-modality. We therefore score against the *best* member.
#[derive(Clone, Debug)]
pub struct Identity {
    pub label: String,
    pub embeddings: Vec<Embedding>,
}

/// Outcome of a gallery query.
#[derive(Clone, Debug, PartialEq)]
pub enum MatchOutcome {
    /// Accepted: label, similarity, and margin over the runner-up (`f32::INFINITY`
    /// when the gallery holds only one identity).
    Match {
        label: String,
        similarity: f32,
        margin: f32,
    },
    /// Best candidate fell below [`MatchConfig::threshold`]. The score is reported so
    /// callers can log near-misses and retune rather than being told only "no".
    BelowThreshold { best: Option<(String, f32)> },
    /// Passed the threshold but two identities were too close to separate.
    Ambiguous {
        first: String,
        second: String,
        margin: f32,
    },
    /// The gallery is empty, or no enrolled embedding had a comparable dimension.
    NoCandidates,
}

/// An in-memory identity gallery supporting enrol and nearest-neighbour match.
///
/// Deliberately a brute-force linear scan. For the thousands-of-identities scale this
/// crate targets, 512-dim dot products are memory-bandwidth bound and vectorise well;
/// an ANN index would add a dependency and an approximation-error failure mode for no
/// measurable win. Revisit past ~1e5 identities.
#[derive(Clone, Debug, Default)]
pub struct Gallery {
    identities: Vec<Identity>,
    config: MatchConfig,
}

impl Gallery {
    pub fn new(config: MatchConfig) -> Self {
        Self {
            identities: Vec::new(),
            config,
        }
    }

    pub fn config(&self) -> &MatchConfig {
        &self.config
    }

    pub fn len(&self) -> usize {
        self.identities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.identities.is_empty()
    }

    pub fn identities(&self) -> &[Identity] {
        &self.identities
    }

    /// Total enrolled embeddings across all identities.
    pub fn embedding_count(&self) -> usize {
        self.identities.iter().map(|i| i.embeddings.len()).sum()
    }

    /// Enrol an embedding under `label`, appending to that identity if it exists.
    pub fn enroll(&mut self, label: impl Into<String>, embedding: Embedding) {
        let label = label.into();
        if let Some(id) = self.identities.iter_mut().find(|i| i.label == label) {
            id.embeddings.push(embedding);
        } else {
            self.identities.push(Identity {
                label,
                embeddings: vec![embedding],
            });
        }
    }

    /// Remove an identity and report whether it was present.
    pub fn remove(&mut self, label: &str) -> bool {
        let before = self.identities.len();
        self.identities.retain(|i| i.label != label);
        self.identities.len() != before
    }

    /// Score `probe` against every identity, best similarity per identity,
    /// sorted descending. Useful for diagnostics and threshold calibration.
    pub fn rank(&self, probe: &Embedding) -> Vec<(String, f32)> {
        let mut scored: Vec<(String, f32)> = self
            .identities
            .iter()
            .filter_map(|id| {
                id.embeddings
                    .iter()
                    // Max over the identity's members: best-matching pose wins.
                    .filter_map(|e| probe.cosine(e))
                    .fold(None, |acc: Option<f32>, s| {
                        Some(acc.map_or(s, |a| a.max(s)))
                    })
                    .map(|s| (id.label.clone(), s))
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }

    /// Identify `probe` under the configured threshold and margin policy.
    pub fn identify(&self, probe: &Embedding) -> MatchOutcome {
        let ranked = self.rank(probe);
        let Some((best_label, best_score)) = ranked.first().cloned() else {
            return MatchOutcome::NoCandidates;
        };

        if best_score < self.config.threshold {
            return MatchOutcome::BelowThreshold {
                best: Some((best_label, best_score)),
            };
        }

        let (margin, runner_up) = match ranked.get(1) {
            Some((l, s)) => (best_score - s, Some(l.clone())),
            None => (f32::INFINITY, None),
        };

        if self.config.min_margin > 0.0 && margin < self.config.min_margin {
            return MatchOutcome::Ambiguous {
                first: best_label,
                second: runner_up.unwrap_or_default(),
                margin,
            };
        }

        MatchOutcome::Match {
            label: best_label,
            similarity: best_score,
            margin,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a normalised embedding from a raw pattern.
    fn emb(v: &[f32]) -> Embedding {
        Embedding::from_raw(v).expect("test vector must be normalisable")
    }

    /// A unit basis vector in `dim` dimensions with `1.0` at `idx`.
    fn basis(idx: usize, dim: usize) -> Embedding {
        let mut v = vec![0.0f32; dim];
        v[idx] = 1.0;
        emb(&v)
    }

    #[test]
    fn from_raw_normalises_to_unit_length() {
        let e = emb(&[3.0, 4.0]);
        let norm: f32 = e.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((e.as_slice()[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn from_raw_rejects_empty_and_zero_and_nan() {
        assert!(Embedding::from_raw(&[]).is_none());
        assert!(Embedding::from_raw(&[0.0, 0.0, 0.0]).is_none());
        assert!(Embedding::from_raw(&[f32::NAN, 1.0]).is_none());
    }

    #[test]
    fn cosine_of_identical_is_one() {
        let e = emb(&[1.0, 2.0, 3.0]);
        assert!((e.cosine(&e).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_opposite_is_minus_one() {
        let a = emb(&[1.0, 0.0]);
        let b = emb(&[-1.0, 0.0]);
        assert!((a.cosine(&b).unwrap() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_is_zero() {
        assert!(basis(0, 4).cosine(&basis(1, 4)).unwrap().abs() < 1e-6);
    }

    #[test]
    fn cosine_is_scale_invariant() {
        // The whole point of normalising: magnitude must not affect similarity.
        let a = emb(&[1.0, 2.0, 3.0]);
        let b = emb(&[10.0, 20.0, 30.0]);
        assert!((a.cosine(&b).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_dimension_mismatch_is_none_not_prefix_compare() {
        assert!(emb(&[1.0, 0.0]).cosine(&emb(&[1.0, 0.0, 0.0])).is_none());
    }

    #[test]
    fn sq_euclidean_matches_two_minus_two_cos() {
        let a = emb(&[1.0, 2.0, 3.0]);
        let b = emb(&[3.0, 1.0, -2.0]);
        let c = a.cosine(&b).unwrap();
        assert!((a.sq_euclidean(&b).unwrap() - (2.0 - 2.0 * c)).abs() < 1e-6);
    }

    #[test]
    fn empty_gallery_reports_no_candidates() {
        let g = Gallery::new(MatchConfig::default());
        assert_eq!(g.identify(&basis(0, 8)), MatchOutcome::NoCandidates);
        assert!(g.is_empty());
    }

    #[test]
    fn enroll_then_identify_exact_match() {
        let mut g = Gallery::new(MatchConfig::default());
        let alice = basis(0, 16);
        g.enroll("alice", alice.clone());
        match g.identify(&alice) {
            MatchOutcome::Match {
                label, similarity, ..
            } => {
                assert_eq!(label, "alice");
                assert!((similarity - 1.0).abs() < 1e-6);
            }
            other => panic!("expected Match, got {other:?}"),
        }
    }

    #[test]
    fn single_identity_gallery_reports_infinite_margin() {
        let mut g = Gallery::new(MatchConfig::default());
        g.enroll("solo", basis(0, 8));
        match g.identify(&basis(0, 8)) {
            MatchOutcome::Match { margin, .. } => assert_eq!(margin, f32::INFINITY),
            other => panic!("expected Match, got {other:?}"),
        }
    }

    #[test]
    fn orthogonal_stranger_falls_below_threshold_and_reports_best() {
        let mut g = Gallery::new(MatchConfig::default());
        g.enroll("alice", basis(0, 8));
        match g.identify(&basis(1, 8)) {
            MatchOutcome::BelowThreshold { best } => {
                let (label, score) = best.expect("near-miss score must be surfaced");
                assert_eq!(label, "alice");
                assert!(score.abs() < 1e-6);
            }
            other => panic!("expected BelowThreshold, got {other:?}"),
        }
    }

    #[test]
    fn enroll_same_label_twice_appends_not_duplicates() {
        let mut g = Gallery::new(MatchConfig::default());
        g.enroll("alice", basis(0, 8));
        g.enroll("alice", basis(1, 8));
        assert_eq!(g.len(), 1, "one identity");
        assert_eq!(g.embedding_count(), 2, "two reference embeddings");
    }

    #[test]
    fn identity_scores_by_best_member_not_average() {
        // alice has a frontal (basis 0) and a profile (basis 1) reference. A probe
        // matching only the profile must still hit 1.0, which averaging would dilute.
        let mut g = Gallery::new(MatchConfig::default());
        g.enroll("alice", basis(0, 8));
        g.enroll("alice", basis(1, 8));
        match g.identify(&basis(1, 8)) {
            MatchOutcome::Match { similarity, .. } => {
                assert!(
                    (similarity - 1.0).abs() < 1e-6,
                    "best-member scoring expected, got {similarity}"
                );
            }
            other => panic!("expected Match, got {other:?}"),
        }
    }

    #[test]
    fn min_margin_flags_ambiguous_twins() {
        // Two enrolled identities nearly equidistant from the probe.
        let cfg = MatchConfig::default()
            .with_threshold(0.1)
            .with_min_margin(0.2);
        let mut g = Gallery::new(cfg);
        g.enroll("twin_a", emb(&[1.0, 0.0]));
        g.enroll("twin_b", emb(&[0.99, 0.14]));
        match g.identify(&emb(&[1.0, 0.07])) {
            MatchOutcome::Ambiguous { margin, .. } => assert!(margin < 0.2),
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn zero_min_margin_disables_ambiguity_check() {
        let cfg = MatchConfig::default()
            .with_threshold(0.1)
            .with_min_margin(0.0);
        let mut g = Gallery::new(cfg);
        g.enroll("twin_a", emb(&[1.0, 0.0]));
        g.enroll("twin_b", emb(&[0.99, 0.14]));
        assert!(matches!(
            g.identify(&emb(&[1.0, 0.07])),
            MatchOutcome::Match { .. }
        ));
    }

    #[test]
    fn rank_is_sorted_descending() {
        let mut g = Gallery::new(MatchConfig::default());
        g.enroll("far", emb(&[0.0, 1.0]));
        g.enroll("near", emb(&[1.0, 0.05]));
        g.enroll("mid", emb(&[0.7, 0.7]));
        let r = g.rank(&emb(&[1.0, 0.0]));
        assert_eq!(r[0].0, "near");
        assert!(r[0].1 >= r[1].1 && r[1].1 >= r[2].1);
    }

    #[test]
    fn remove_identity_reports_presence() {
        let mut g = Gallery::new(MatchConfig::default());
        g.enroll("alice", basis(0, 8));
        assert!(g.remove("alice"));
        assert!(!g.remove("alice"), "second removal must report absent");
        assert!(g.is_empty());
    }

    #[test]
    fn mismatched_dim_gallery_yields_no_candidates() {
        // A gallery built with another model's embeddings must not silently match.
        let mut g = Gallery::new(MatchConfig::default());
        g.enroll("alice", basis(0, 128));
        assert_eq!(g.identify(&basis(0, 512)), MatchOutcome::NoCandidates);
    }

    #[test]
    fn threshold_boundary_is_inclusive() {
        let mut g = Gallery::new(MatchConfig::default().with_threshold(0.5));
        g.enroll("alice", emb(&[1.0, 0.0]));
        // Construct a probe with cosine exactly 0.5 (60 degrees).
        let probe = emb(&[0.5, (3.0f32).sqrt() / 2.0]);
        let sim = probe.cosine(&emb(&[1.0, 0.0])).unwrap();
        assert!((sim - 0.5).abs() < 1e-6, "sanity: sim={sim}");
        assert!(
            matches!(g.identify(&probe), MatchOutcome::Match { .. }),
            "similarity == threshold must be accepted"
        );
    }

    #[test]
    fn arcface_dim_constant_is_512() {
        assert_eq!(ARCFACE_EMBEDDING_DIM, 512);
    }
}
