//! Multi-algorithm ensemble face detector.
//!
//! Fuses detections from N independent detectors into a single higher-
//! precision set. Boxes whose pairwise IoU exceeds `iou_threshold` are
//! treated as the same face, the group bbox is a weighted average of
//! its members, and the confidence is a weighted average of the member
//! scores. The cluster must have at least `min_votes` members to be
//! emitted, which is the cheap consensus gate — a Haar-only hit is one
//! vote, a Haar + luminance agreement on the same window is two.
//!
//! ## Why this matters
//!
//! The three classical zero-dep detectors in this crate have
//! **complementary** failure modes (see memory 2026-08-14, 2026-09-08):
//!
//! - **Haar** (Viola-Jones): reliable on real frontal faces, fires
//!   rarely on synthetic / low-contrast inputs, can miss rotated or
//!   partially-occluded faces.
//! - **luminance** (band + symmetry): fires on the canonical
//!   forehead/eye-band/chin signature but rejects real faces where
//!   the band gate is too strict (lena / biden / demo_face_256 all
//!   score 0 with the default config).
//! - **CNN** (template weights): over-fires with the bundled
//!   hand-crafted weights — 1000+ detections on lena with conf=1.0 —
//!   so by itself the CNN is a precision disaster. As a *vote*
//!   inside an ensemble, the consensus filter removes most of that.
//!
//! Pairwise ensemble of any two of them gives a free recall floor at
//! the union of their detections and a precision floor at the
//! intersection; the weighted average refines the box.
//!
//! ## Weights
//!
//! Per-algorithm reliability is set at call time as `source_weight`,
//! defaulting to 1.0. Haar is given 1.0 in the recipes below; CNN
//! template weights are down-weighted to 0.4 because their score
//! distribution is uncalibrated; luminance gets 0.7 because its
//! geometric-mean score is informative but not as precise as Haar's
//! cascade score. The numbers are tunable per-deployment via
//! [`EnsembleConfig::source_weights`].

use crate::face::{iou, Detection};

/// A single detection tagged with the algorithm that produced it.
///
/// The `source` string is the same name returned by
/// [`crate::face_detector::FaceDetector::name`] (e.g. `"haar"`,
/// `"luminance"`, `"cnn"`) and is preserved on the fused output so
/// callers can show the consensus breakdown ("haar + luminance agree
/// at x=…") on the platform compare card.
#[derive(Clone, Debug)]
pub struct TaggedDetection {
    pub detection: Detection,
    pub source: &'static str,
    /// Per-algorithm reliability weight in `[0, 1]`. 0.0 silences the
    /// detector entirely from the ensemble; 1.0 is the default.
    pub source_weight: f32,
}

impl TaggedDetection {
    pub fn new(detection: Detection, source: &'static str, source_weight: f32) -> Self {
        Self {
            detection,
            source,
            source_weight: source_weight.max(0.0),
        }
    }

    /// Sort key used to drive the greedy cluster assignment: the
    /// strongest vote casts first, so weaker votes can only attach
    /// to a cluster that a stronger algorithm already seeded.
    #[inline]
    fn rank(&self) -> f32 {
        self.source_weight * self.detection.score.max(0.0)
    }
}

/// Tunable knobs for the ensemble fuser.
#[derive(Clone, Debug)]
pub struct EnsembleConfig {
    /// IoU above this groups two detections as the same face.
    pub iou_threshold: f32,
    /// Minimum cluster size to emit a fused detection.
    pub min_votes: usize,
    /// Per-source weight overrides; key is the algorithm name. A
    /// missing entry defaults to `1.0`. Use this to silence an
    /// unreliable detector (set weight 0.0) without rebuilding the
    /// input list.
    pub source_weights: std::collections::HashMap<&'static str, f32>,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            // 0.3 matches the per-algorithm NMS threshold used by every
            // detector in the crate, so two detectors agreeing on the
            // same window land in the same cluster.
            iou_threshold: 0.3,
            min_votes: 1,
            source_weights: std::collections::HashMap::new(),
        }
    }
}

impl EnsembleConfig {
    /// Look up the source weight with the default fallback.
    pub fn weight(&self, source: &'static str) -> f32 {
        self.source_weights
            .get(source)
            .copied()
            .unwrap_or(1.0)
            .max(0.0)
    }
}

#[derive(Clone, Debug, Default)]
struct Cluster {
    members: Vec<TaggedDetection>,
    /// Sum of effective weights — cached so we can normalise the bbox
    /// average in one pass.
    total_w: f32,
}

impl Cluster {
    /// Push a new member and keep the running weight sum up to date.
    fn push(&mut self, m: TaggedDetection) {
        let w = m.rank();
        self.total_w += w;
        self.members.push(m);
    }

    /// Centroid detection of this cluster: weighted average of every
    /// member's `(x, y, w, h)` and arithmetic-mean confidence.
    fn fused(&self) -> Detection {
        let n = self.members.len();
        if self.total_w <= 0.0 {
            // Degenerate (all weights zero). Fall back to a plain mean
            // so we still emit a sane box.
            let mut bx = 0.0f32;
            let mut by = 0.0f32;
            let mut bw = 0.0f32;
            let mut bh = 0.0f32;
            let mut bs = 0.0f32;
            for m in &self.members {
                bx += m.detection.x as f32;
                by += m.detection.y as f32;
                bw += m.detection.w as f32;
                bh += m.detection.h as f32;
                bs += m.detection.score;
            }
            let inv = 1.0 / n as f32;
            return Detection {
                x: (bx * inv).round() as usize,
                y: (by * inv).round() as usize,
                w: (bw * inv).round().max(1.0) as usize,
                h: (bh * inv).round().max(1.0) as usize,
                score: bs * inv,
            };
        }
        let mut bx = 0.0f32;
        let mut by = 0.0f32;
        let mut bw = 0.0f32;
        let mut bh = 0.0f32;
        let mut bs = 0.0f32;
        for m in &self.members {
            let w = m.rank();
            let inv = w / self.total_w;
            bx += inv * m.detection.x as f32;
            by += inv * m.detection.y as f32;
            bw += inv * m.detection.w as f32;
            bh += inv * m.detection.h as f32;
            bs += inv * m.detection.score;
        }
        Detection {
            x: bx.round() as usize,
            y: by.round() as usize,
            w: bw.round().max(1.0) as usize,
            h: bh.round().max(1.0) as usize,
            score: bs,
        }
    }

    /// Names of every source that contributed a member. Useful for the
    /// consensus-breakdown UI: `"haar+luminance"` on a 2-vote cluster.
    fn sources(&self) -> String {
        let mut names: Vec<&'static str> = self.members.iter().map(|m| m.source).collect();
        names.sort_unstable();
        names.dedup();
        names.join("+")
    }
}

/// Outcome of a single fused cluster — the centroid detection plus the
/// per-source membership so the platform compare endpoint can show
/// "haar + luminance agreed on this face".
#[derive(Clone, Debug)]
pub struct FusedCluster {
    pub detection: Detection,
    pub sources: String,
    pub votes: usize,
}

/// Fuse N sets of detections into a single higher-precision set.
///
/// The algorithm is a single-pass greedy cluster:
/// 1. Sort every input by `source_weight * score` (descending) so the
///    strongest votes seed the clusters first.
/// 2. For each input, find the existing cluster with the highest IoU
///    against it; if that IoU exceeds `iou_threshold`, attach the
///    input to that cluster, otherwise start a new one.
/// 3. Emit a cluster as a [`FusedCluster`] iff `votes >= min_votes`.
///
/// This is a deterministic, allocation-light pass: O(N · C) in the
/// number of inputs and clusters, no hash table, no recursion. For
/// the small N (typically < 100) typical of zero-dep detection on a
/// single image this is dominated by the IoU arithmetic.
pub fn fuse(inputs: Vec<TaggedDetection>, config: &EnsembleConfig) -> Vec<FusedCluster> {
    if inputs.is_empty() {
        return Vec::new();
    }
    // Apply per-source weight overrides (silenced detectors drop to
    // 0.0, real ones keep their declared weight). Done once here so
    // the cluster assignment reads the same `rank()` the user expects.
    let mut sorted: Vec<TaggedDetection> = inputs
        .into_iter()
        .map(|mut m| {
            let declared = m.source_weight;
            let override_w = config.weight(m.source);
            // The smaller of the caller's weight and the config
            // override wins; this lets the platform dial down a noisy
            // detector without the detector having to know about it.
            m.source_weight = declared.min(override_w).max(0.0);
            m
        })
        .filter(|m| m.source_weight > 0.0 && m.detection.w > 0 && m.detection.h > 0)
        .collect();
    sorted.sort_unstable_by(|a, b| b.rank().total_cmp(&a.rank()));

    let mut clusters: Vec<Cluster> = Vec::new();
    for m in sorted {
        let mut best_idx: Option<usize> = None;
        let mut best_iou = config.iou_threshold; // strict-greater comparison below
        for (i, c) in clusters.iter().enumerate() {
            // A cluster's representative box is its centroid at the
            // current moment; comparing against the centroid (rather
            // than every member) keeps the inner loop O(C) instead of
            // O(C · cluster_size) and is monotone enough for the
            // greedy pass to converge.
            let centroid = c.fused();
            let cur = iou(&centroid, &m.detection);
            if cur > best_iou {
                best_iou = cur;
                best_idx = Some(i);
            }
        }
        match best_idx {
            Some(i) => clusters[i].push(m),
            None => {
                let mut c = Cluster::default();
                c.push(m);
                clusters.push(c);
            }
        }
    }

    clusters
        .into_iter()
        .filter(|c| c.members.len() >= config.min_votes)
        .map(|c| {
            let votes = c.members.len();
            let sources = c.sources();
            let detection = c.fused();
            FusedCluster {
                detection,
                sources,
                votes,
            }
        })
        .collect()
}

/// Convenience wrapper that returns just the centroids, dropping the
/// cluster membership metadata. Equivalent to
/// `fuse(...).into_iter().map(|c| c.detection).collect()`.
pub fn fuse_detections(inputs: Vec<TaggedDetection>, config: &EnsembleConfig) -> Vec<Detection> {
    fuse(inputs, config)
        .into_iter()
        .map(|c| c.detection)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: usize, y: usize, w: usize, h: usize, score: f32) -> Detection {
        Detection { x, y, w, h, score }
    }

    #[test]
    fn fuse_empty_is_empty() {
        assert!(fuse(Vec::new(), &EnsembleConfig::default()).is_empty());
    }

    #[test]
    fn fuse_single_member_emits_one_cluster() {
        let out = fuse(
            vec![TaggedDetection::new(det(10, 20, 30, 40, 0.9), "haar", 1.0)],
            &EnsembleConfig::default(),
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].votes, 1);
        assert_eq!(out[0].sources, "haar");
        assert_eq!(
            (
                out[0].detection.x,
                out[0].detection.y,
                out[0].detection.w,
                out[0].detection.h
            ),
            (10, 20, 30, 40)
        );
    }

    #[test]
    fn fuse_two_overlapping_becomes_one_cluster() {
        // Two boxes with IoU ~ 0.5 — they must merge.
        let a = TaggedDetection::new(det(0, 0, 100, 100, 0.7), "haar", 1.0);
        let b = TaggedDetection::new(det(50, 0, 100, 100, 0.9), "luminance", 1.0);
        let out = fuse(vec![a, b], &EnsembleConfig::default());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].votes, 2);
        assert_eq!(out[0].sources, "haar+luminance");
    }

    #[test]
    fn fuse_two_disjoint_stays_two() {
        let a = TaggedDetection::new(det(0, 0, 50, 50, 0.7), "haar", 1.0);
        let b = TaggedDetection::new(det(500, 500, 50, 50, 0.9), "luminance", 1.0);
        let out = fuse(vec![a, b], &EnsembleConfig::default());
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn min_votes_filters_singleton_algorithms() {
        let a = TaggedDetection::new(det(0, 0, 50, 50, 0.5), "haar", 1.0);
        let b = TaggedDetection::new(det(500, 500, 50, 50, 0.5), "luminance", 1.0);
        let cfg = EnsembleConfig {
            min_votes: 2,
            ..EnsembleConfig::default()
        };
        // Two disjoint hits, no cluster has 2 members → all dropped.
        assert!(fuse(vec![a, b], &cfg).is_empty());
    }

    #[test]
    fn source_weight_zero_silences_detector() {
        let a = TaggedDetection::new(det(0, 0, 50, 50, 0.9), "haar", 1.0);
        let b = TaggedDetection::new(det(0, 0, 50, 50, 0.5), "cnn", 0.0); // silenced
        let out = fuse(vec![a, b], &EnsembleConfig::default());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].votes, 1);
        assert_eq!(out[0].sources, "haar");
    }

    #[test]
    fn config_weight_override_silences_detector() {
        let a = TaggedDetection::new(det(0, 0, 50, 50, 0.9), "haar", 1.0);
        let b = TaggedDetection::new(det(0, 0, 50, 50, 0.5), "cnn", 0.4);
        let mut cfg = EnsembleConfig::default();
        cfg.source_weights.insert("cnn", 0.0);
        let out = fuse(vec![a, b], &cfg);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].sources, "haar");
    }

    #[test]
    fn stronger_detector_anchors_centroid() {
        // Two close boxes; Haar is heavy (1.0) and slightly offset,
        // luminance is light (0.4) and a little larger. Centroid must
        // bias toward Haar. The boxes overlap substantially (IoU ~ 0.6)
        // so they cluster together.
        let haar = TaggedDetection::new(det(10, 10, 40, 40, 0.9), "haar", 1.0);
        let lum = TaggedDetection::new(det(15, 15, 50, 50, 0.9), "luminance", 0.4);
        let out = fuse(vec![haar, lum], &EnsembleConfig::default());
        assert_eq!(out.len(), 1, "two overlapping boxes must cluster");
        // Haar contribution: 1.0 * 0.9 = 0.9. Luminance: 0.4 * 0.9 = 0.36.
        // Centroid x = 10 * (0.9/1.26) + 15 * (0.36/1.26) ≈ 7.14 + 4.29 ≈ 11.4.
        // (Luminance's x=15 contributes less than Haar's x=10 because Haar's
        // weight is heavier.)
        let cx = out[0].detection.x;
        assert!(
            cx < 13,
            "centroid should lean toward Haar's x=10 (lum x=15), got {cx}"
        );
    }

    #[test]
    fn degenerate_zero_weight_inputs_are_dropped() {
        let a = TaggedDetection::new(det(0, 0, 50, 50, 0.0), "haar", 0.0);
        let b = TaggedDetection::new(det(0, 0, 50, 50, 0.0), "cnn", 0.0);
        // Both zero-weighted → after filter, input list is empty.
        assert!(fuse(vec![a, b], &EnsembleConfig::default()).is_empty());
    }

    #[test]
    fn degenerate_zero_area_box_is_dropped() {
        let a = TaggedDetection::new(det(0, 0, 0, 50, 0.9), "haar", 1.0);
        let b = TaggedDetection::new(det(50, 50, 50, 0, 0.9), "luminance", 1.0);
        let out = fuse(vec![a, b], &EnsembleConfig::default());
        assert!(out.is_empty(), "zero-area boxes must not cluster");
    }
}
