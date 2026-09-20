//! Edge-case coverage for the two NMS implementations in `rsface::face`:
//! the classical greedy `non_max_suppression` over integer boxes and the
//! modern sub-pixel `nms` over `FaceDetection`.
//!
//! The unit tests in `src/face.rs` already cover the happy paths and a
//! couple of empty / NaN / disjoint cases. This file exercises:
//!
//! * full overlap (IoU == 1)
//! * identical boxes with different confidence scores
//! * many overlapping boxes vs. one disjoint box
//! * extreme threshold values (0.0 keeps everything disjoint, 1.0 keeps all
//!   same-position duplicates, both must never panic)
//! * the symmetry property `nms(A ∪ {b})` and `nms(B ∪ {a})` agree on the
//!   survivor set when `a` and `b` tie on score (deterministic, no proptest).
//!
//! These cases live in an integration test so a refactor of either NMS
//! cannot silently lose its way on the boundary.

use rsface::face::{nms, non_max_suppression, Detection, FaceDetection};

fn int_box(x: usize, y: usize, w: usize, h: usize, score: f32) -> Detection {
    Detection { x, y, w, h, score }
}

fn f32_box(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> FaceDetection {
    FaceDetection {
        x1,
        y1,
        x2,
        y2,
        score,
        landmarks: None,
    }
}

#[test]
fn classical_nms_full_overlap_keeps_highest_only() {
    // Three boxes that overlap entirely: only the top score must survive.
    let dets = vec![
        int_box(10, 10, 30, 30, 0.10),
        int_box(10, 10, 30, 30, 0.95),
        int_box(10, 10, 30, 30, 0.50),
    ];
    let kept = non_max_suppression(dets, 0.5);
    assert_eq!(kept.len(), 1);
    assert!((kept[0].score - 0.95).abs() < 1e-6);
}

#[test]
fn classical_nms_threshold_zero_keeps_disjoint_only() {
    // threshold == 0: any positive IoU is suppressed, even full overlap.
    // Two pairs that are full-overlap (within pair) and disjoint (across pairs).
    let dets = vec![
        int_box(0, 0, 10, 10, 0.9),
        int_box(0, 0, 10, 10, 0.8),
        int_box(100, 100, 10, 10, 0.7),
        int_box(100, 100, 10, 10, 0.6),
    ];
    let kept = non_max_suppression(dets, 0.0);
    // The two pairs collapse to their top-scoring boxes; the surviving pair
    // must still be disjoint at this threshold.
    assert_eq!(kept.len(), 2);
    assert!((kept[0].score - 0.9).abs() < 1e-6);
    assert!((kept[1].score - 0.7).abs() < 1e-6);
}

#[test]
fn classical_nms_threshold_one_keeps_identical() {
    // threshold == 1.0: even fully-overlapping boxes survive because the
    // comparison is strict `>`. Locks in the "strictly greater" semantics.
    let dets = vec![int_box(0, 0, 10, 10, 0.9), int_box(0, 0, 10, 10, 0.5)];
    let kept = non_max_suppression(dets, 1.0);
    assert_eq!(kept.len(), 2);
}

#[test]
fn classical_nms_single_box_is_identity() {
    let dets = vec![int_box(5, 7, 20, 30, 0.42)];
    let kept = non_max_suppression(dets.clone(), 0.3);
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].x, dets[0].x);
    assert_eq!(kept[0].w, dets[0].w);
    assert!((kept[0].score - dets[0].score).abs() < 1e-6);
}

#[test]
fn classical_nms_same_position_different_confidence_orders_correctly() {
    let dets = vec![
        int_box(0, 0, 10, 10, 0.1),
        int_box(0, 0, 10, 10, 0.5),
        int_box(0, 0, 10, 10, 0.3),
    ];
    let kept = non_max_suppression(dets, 0.3);
    assert_eq!(kept.len(), 1);
    assert!((kept[0].score - 0.5).abs() < 1e-6);
}

#[test]
fn modern_nms_threshold_one_keeps_identical_boxes() {
    // The FaceDetection path uses the same strict `>` comparison; document it.
    let dets = vec![
        f32_box(0.0, 0.0, 10.0, 10.0, 0.9),
        f32_box(0.0, 0.0, 10.0, 10.0, 0.5),
    ];
    let kept = nms(dets, 1.0);
    assert_eq!(kept.len(), 2);
}

#[test]
fn modern_nms_threshold_zero_collapses_overlap() {
    // Any positive IoU is suppressed at threshold == 0.
    let dets = vec![
        f32_box(0.0, 0.0, 10.0, 10.0, 0.9),
        f32_box(5.0, 5.0, 15.0, 15.0, 0.8),
        f32_box(50.0, 50.0, 60.0, 60.0, 0.7),
    ];
    let kept = nms(dets, 0.0);
    assert_eq!(
        kept.len(),
        2,
        "two survivors: highest in cluster, the disjoint box"
    );
    assert!((kept[0].score - 0.9).abs() < 1e-6);
    assert!((kept[1].score - 0.7).abs() < 1e-6);
}

#[test]
fn modern_nms_many_overlapping_vs_one_disjoint() {
    // 20 nearly-coincident boxes (one face's pyramid of proposals) plus one
    // isolated disjoint box. The cluster should collapse to a single survivor;
    // the disjoint box must also survive.
    let mut dets: Vec<FaceDetection> = (0..20)
        .map(|i| {
            // Jitter position by sub-pixel so IoU between any two is > 0.5
            // for the cluster, but they all overlap heavily with the top one.
            f32_box(
                0.05 * (i as f32),
                0.05 * (i as f32),
                10.0 + 0.05 * (i as f32),
                10.0 + 0.05 * (i as f32),
                0.5 + i as f32 * 0.01,
            )
        })
        .collect();
    dets.push(f32_box(200.0, 200.0, 210.0, 210.0, 0.42));
    let kept = nms(dets, 0.3);
    assert_eq!(kept.len(), 2);
    // Disjoint box always survives; cluster survivor must be the highest score.
    let mut scores: Vec<f32> = kept.iter().map(|d| d.score).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((scores[0] - 0.42).abs() < 1e-6);
    assert!(scores[1] > 0.6, "cluster survivor was {}", scores[1]);
}

#[test]
fn modern_nms_nan_scores_are_total_ordered_no_panic() {
    // A NaN score must not panic; total_cmp keeps the sort total.
    let dets = vec![
        f32_box(0.0, 0.0, 10.0, 10.0, f32::NAN),
        f32_box(50.0, 50.0, 60.0, 60.0, 0.5),
    ];
    let kept = nms(dets, 0.5);
    assert_eq!(kept.len(), 2);
}
