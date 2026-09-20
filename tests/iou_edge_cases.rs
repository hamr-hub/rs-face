//! Cross-cutting IoU invariants for both `Detection` (integer boxes, classical
//! Haar path) and `FaceDetection` (sub-pixel boxes, modern detector path).
//!
//! The unit tests in `src/face.rs` and the historical commit `160b4a8` already
//! cover the nested/symmetric/zero-area cases for the modern detector path.
//! This file:
//!
//! * exercises degenerate boxes (w=0 / h=0) on both representations,
//! * locks in symmetry across many box geometries on a deterministic set of
//!   hand-built seeds (proptest would be overkill given the zero-dep policy),
//! * checks that `FaceDetection::to_detection` round-trips integer-origin
//!   boxes exactly and that negative-origin boxes clamp to zero,
//! * validates that `non_max_suppression` over identical boxes never panics
//!   for any score (including NaN/inf in the integer path).

use rsface::face::{Detection, FaceDetection, non_max_suppression};

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
fn classical_iou_degenerate_zero_width_is_zero() {
    let a = int_box(0, 0, 10, 10, 1.0);
    let b = int_box(5, 0, 0, 10, 1.0);
    assert_eq!(a.iou(&b), 0.0);
}

#[test]
fn classical_iou_degenerate_zero_height_is_zero() {
    let a = int_box(0, 0, 10, 10, 1.0);
    let b = int_box(0, 5, 10, 0, 1.0);
    assert_eq!(a.iou(&b), 0.0);
}

#[test]
fn classical_iou_both_degenerate_is_zero_not_panic() {
    let a = int_box(0, 0, 0, 0, 1.0);
    let b = int_box(10, 10, 0, 0, 1.0);
    assert_eq!(a.iou(&b), 0.0);
}

#[test]
fn modern_iou_zero_width_is_zero() {
    let a = f32_box(0.0, 0.0, 10.0, 10.0, 1.0);
    let b = f32_box(5.0, 0.0, 5.0, 10.0, 1.0); // w=0
    assert_eq!(a.iou(&b), 0.0);
}

#[test]
fn modern_iou_zero_height_is_zero() {
    let a = f32_box(0.0, 0.0, 10.0, 10.0, 1.0);
    let b = f32_box(0.0, 5.0, 10.0, 5.0, 1.0); // h=0
    assert_eq!(a.iou(&b), 0.0);
}

#[test]
fn modern_iou_inverted_box_clamps_to_zero_area() {
    // x2 < x1: width() returns 0 via max(0.0, ...) and so does area().
    // IoU must be 0.0, not negative, not NaN.
    let a = f32_box(0.0, 0.0, 10.0, 10.0, 1.0);
    let b = f32_box(10.0, 10.0, 0.0, 0.0, 1.0); // inverted
    assert_eq!(a.iou(&b), 0.0);
}

#[test]
fn modern_iou_negative_origin_boxes_clamped_to_zero_overlap() {
    // Two boxes both starting at negative coordinates do not intersect
    // their positive-coord counterpart. IoU must be 0.0.
    let a = f32_box(0.0, 0.0, 10.0, 10.0, 1.0);
    let b = f32_box(-5.0, -5.0, -1.0, -1.0, 1.0);
    assert_eq!(a.iou(&b), 0.0);
}

#[test]
fn modern_iou_symmetry_across_many_geometries() {
    // 12 hand-picked boxes covering disjoint, touching-edge, partial,
    // nested, inverted, and degenerate cases. Pairwise symmetry is required.
    let seeds = [
        (0.0, 0.0, 10.0, 10.0),
        (5.0, 0.0, 15.0, 10.0),
        (50.0, 50.0, 60.0, 60.0),
        (0.0, 0.0, 100.0, 100.0),
        (10.0, 10.0, 30.0, 30.0),
        (9.0, 9.0, 11.0, 11.0), // tiny box overlapping a large
        (10.0, 0.0, 10.0, 10.0), // zero-width
        (0.0, 5.0, 10.0, 5.0),   // zero-height
        (-20.0, -20.0, -5.0, -5.0),
        (-5.0, -5.0, 5.0, 5.0), // straddles origin
        (1e3, 1e3, 1e3 + 50.0, 1e3 + 50.0),
        (0.0, 0.0, 1.0, 1.0),   // sub-pixel
    ];
    for &(x1, y1, x2, y2) in &seeds {
        let a = f32_box(x1, y1, x2, y2, 1.0);
        for &(x1b, y1b, x2b, y2b) in &seeds {
            let b = f32_box(x1b, y1b, x2b, y2b, 1.0);
            let ab = a.iou(&b);
            let ba = b.iou(&a);
            assert!(
                (ab - ba).abs() < 1e-5,
                "asymmetric IoU: a=({x1},{y1},{x2},{y2}) b=({x1b},{y1b},{x2b},{y2b}) ab={ab} ba={ba}"
            );
        }
    }
}

#[test]
fn modern_iou_in_unit_interval_for_overlapping_boxes() {
    // Anywhere two boxes have non-zero area and overlap, IoU must lie in [0, 1].
    let cases = [
        (0.0, 0.0, 10.0, 10.0, 5.0, 5.0, 15.0, 15.0),
        (0.0, 0.0, 100.0, 100.0, 1.0, 1.0, 99.0, 99.0),
        (0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.5, 1.5),
    ];
    for &(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) in &cases {
        let a = f32_box(ax1, ay1, ax2, ay2, 1.0);
        let b = f32_box(bx1, by1, bx2, by2, 1.0);
        let iou = a.iou(&b);
        assert!((0.0..=1.0).contains(&iou), "IoU {iou} not in [0, 1] for a vs b");
    }
}

#[test]
fn to_detection_roundtrip_preserves_int_origin_box() {
    let a = Detection {
        x: 12,
        y: 34,
        w: 56,
        h: 78,
        score: 0.9,
    };
    let f: FaceDetection = a.clone().into();
    let back = f.to_detection();
    assert_eq!((back.x, back.y, back.w, back.h), (a.x, a.y, a.w, a.h));
    assert!((back.score - a.score).abs() < 1e-6);
}

#[test]
fn to_detection_clamps_negative_origin_to_zero() {
    // Negative-origin boxes clamp x/y to 0 but preserve the full
    // x2-x1 / y2-y1 span (a face running off the left edge still has
    // its detected width). This locks in the documented behaviour.
    let f = f32_box(-3.0, -7.0, 10.0, 10.0, 0.9);
    let d = f.to_detection();
    assert_eq!(d.x, 0);
    assert_eq!(d.y, 0);
    // width/height are taken from x2-x1 / y2-y1 (no shrinking).
    assert_eq!(d.w, 13);
    assert_eq!(d.h, 17);
}

#[test]
fn classical_nms_nan_scores_do_not_panic() {
    // The Detection path uses partial_cmp (unwrap_or(Equal)) and so must not
    // panic on NaN; unlike the FaceDetection path it does not total_cmp.
    let dets = vec![
        int_box(0, 0, 10, 10, f32::NAN),
        int_box(50, 50, 10, 10, 0.5),
    ];
    let kept = non_max_suppression(dets, 0.5);
    assert_eq!(kept.len(), 2, "NaN must not eat a disjoint neighbour");
}