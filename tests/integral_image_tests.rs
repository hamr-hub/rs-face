//! Sanity / boundary tests for the integral-image table (`rsface::integral`).
//!
//! These tests run against a **naive** reference implementation that
//! recomputes every prefix sum by brute force, and require the optimized
//! `IntegralImage::rect_sum` to produce bit-identical answers. The goal is
//! to catch any future refactor that breaks the inclusion–exclusion identity
//! on degenerate input (zero-size images, single-column, single-pixel,
//! checkerboard) before it can silently corrupt a Haar score downstream.
//!
//! Requires the `detector-haar` feature (the integral table is part of the
//! Haar cascade stack and is not compiled under `--no-default-features`).

use rsface::image::GrayImage;
use rsface::integral::{IntegralImage, SquaredIntegralImage};

/// Brute-force O(W*H) recomputation of the rectangular sum. The whole point
/// of the integral image is to turn this into O(1), but writing it again
/// from scratch is the cheapest possible oracle.
fn ref_rect_sum(img: &GrayImage, x1: usize, y1: usize, x2: usize, y2: usize) -> u64 {
    let w = img.width();
    let mut s: u64 = 0;
    for y in y1..y2.min(img.height()) {
        for x in x1..x2.min(w) {
            s += img[(x, y)] as u64;
        }
    }
    s
}

/// Brute-force squared sum. Same oracle role as `ref_rect_sum`, just on the
/// sum-of-squares integral.
fn ref_rect_sum_sq(img: &GrayImage, x1: usize, y1: usize, x2: usize, y2: usize) -> u64 {
    let mut s: u64 = 0;
    for y in y1..y2.min(img.height()) {
        for x in x1..x2.min(img.width()) {
            let v = img[(x, y)] as u64;
            s += v * v;
        }
    }
    s
}

fn constant(w: usize, h: usize, v: u8) -> GrayImage {
    GrayImage::from_vec(vec![v; w * h], w, h)
}

#[test]
fn all_zeros_integral_rect_sum_is_zero() {
    let img = constant(16, 12, 0);
    let ii = IntegralImage::from_gray(&img);
    for y1 in 0..=12 {
        for y2 in y1..=12 {
            for x1 in 0..=16 {
                for x2 in x1..=16 {
                    assert_eq!(ii.rect_sum(x1, y1, x2, y2), 0);
                }
            }
        }
    }
}

#[test]
fn all_max_integral_matches_reference_on_random_rects() {
    // 255 everywhere: every rect sum is 255*w*h. The reference and the
    // optimized build must agree exactly (and the u32 path must hold even
    // at the largest size we exercise).
    let img = constant(64, 48, 255);
    let ii = IntegralImage::from_gray(&img);
    assert!(!ii.is_wide(), "64*48*255 is nowhere near u32 overflow");
    for &(x1, y1, x2, y2) in &[
        (0, 0, 64, 48),
        (0, 0, 1, 1),
        (10, 5, 50, 40),
        (1, 1, 2, 2),
        (63, 47, 64, 48), // 1-pixel corner
    ] {
        assert_eq!(
            ii.rect_sum(x1, y1, x2, y2),
            ref_rect_sum(&img, x1, y1, x2, y2)
        );
    }
}

#[test]
fn checkerboard_integral_matches_reference_on_every_rect() {
    // Alternating 0/255 — stresses the inclusion-exclusion identity on
    // boxes whose corner sum cancels on odd widths.
    let (w, h) = (32, 24);
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img[(x, y)] = if (x + y) % 2 == 0 { 255 } else { 0 };
        }
    }
    let ii = IntegralImage::from_gray(&img);
    let rects = [
        (0, 0, w, h),
        (1, 1, w - 1, h - 1),
        (0, 0, 1, 1),
        (3, 5, 17, 11),
        (w / 2, h / 2, w, h),
    ];
    for &(x1, y1, x2, y2) in &rects {
        assert_eq!(
            ii.rect_sum(x1, y1, x2, y2),
            ref_rect_sum(&img, x1, y1, x2, y2),
            "mismatch at ({x1},{y1},{x2},{y2})"
        );
    }
}

#[test]
fn single_pixel_integral_returns_pixel_value() {
    let img = GrayImage::from_vec(vec![123], 1, 1);
    let ii = IntegralImage::from_gray(&img);
    assert_eq!(ii.rect_sum(0, 0, 1, 1), 123);
}

#[test]
fn single_column_integral_matches_reference() {
    // 1xN column — exercises the edge case where x2 == x1+1 on every row.
    let img = GrayImage::from_vec(vec![10, 20, 30, 40, 50, 60, 70, 80], 1, 8);
    let ii = IntegralImage::from_gray(&img);
    for y1 in 0..=8 {
        for y2 in y1..=8 {
            assert_eq!(
                ii.rect_sum(0, y1, 1, y2),
                ref_rect_sum(&img, 0, y1, 1, y2),
                "y1={y1} y2={y2}"
            );
        }
    }
}

#[test]
fn empty_rect_returns_zero() {
    let img = constant(8, 8, 200);
    let ii = IntegralImage::from_gray(&img);
    assert_eq!(ii.rect_sum(0, 0, 0, 0), 0);
    assert_eq!(ii.rect_sum(3, 3, 3, 5), 0);
    assert_eq!(ii.rect_sum(2, 4, 5, 4), 0);
    assert_eq!(ii.rect_sum(3, 3, 3, 3), 0);
}

#[test]
fn clamped_rect_does_not_panic_on_out_of_bounds() {
    // Asking for a rect that hangs off the image is silently clamped to the
    // image bounds; it must not panic and must equal the reference.
    let img = constant(16, 12, 100);
    let ii = IntegralImage::from_gray(&img);
    // Asking far past the right/bottom edges:
    let s1 = ii.rect_sum(0, 0, 16, 12);
    let s2 = ii.rect_sum(0, 0, 100, 100);
    assert_eq!(s1, s2, "out-of-bounds must clamp");
    assert_eq!(s1, ref_rect_sum(&img, 0, 0, 16, 12));
}

#[test]
fn wide_path_taken_for_very_large_image_dimensions() {
    // W*H*255 > u32::MAX is the threshold; pick a 4500x4500 image
    // (4500*4500*255 = 5.16e9 > 4.29e9). Pool is bypassed on this path
    // so the test exercises the non-pooled build, too.
    let (w, h) = (4500, 4500);
    let img = constant(w, h, 1);
    let ii = IntegralImage::from_gray(&img);
    assert!(
        ii.is_wide(),
        "4500*4500*255 must overflow u32 and trigger Wide path"
    );
    // full image sum = w*h*1
    assert_eq!(ii.rect_sum(0, 0, w, h), (w as u64) * (h as u64));
}

#[test]
fn wide_path_rect_sum_matches_reference_for_large_image() {
    // Spot-check a sub-rect against the reference; the wide u64 path
    // must not drift.
    let (w, h) = (4500, 4500);
    let img = constant(w, h, 7);
    let ii = IntegralImage::from_gray(&img);
    assert!(ii.is_wide());
    // Top-left 100x100 block:
    assert_eq!(ii.rect_sum(0, 0, 100, 100), 7 * 100 * 100);
    // Arbitrary interior:
    assert_eq!(ii.rect_sum(2000, 2000, 2500, 2500), 7 * 500 * 500);
}

#[test]
fn squared_integral_matches_reference_for_max_image() {
    // SquaredIntegralImage is the variance pre-filter oracle: spot-check
    // a few rectangles against the brute-force sum-of-squares.
    let (w, h) = (16, 12);
    let img = constant(w, h, 200);
    let sq = SquaredIntegralImage::from_gray(&img);
    let rects = [(0, 0, w, h), (0, 0, 1, 1), (3, 5, 10, 9), (15, 11, 16, 12)];
    for &(x1, y1, x2, y2) in &rects {
        assert_eq!(
            sq.rect_sum_sq(x1, y1, x2, y2),
            ref_rect_sum_sq(&img, x1, y1, x2, y2),
            "sq mismatch at ({x1},{y1},{x2},{y2})"
        );
    }
}

#[test]
fn integral_at_zero_zero_is_zero_padding() {
    // First row / first column of the integral table is the zero-pad by
    // construction. At(0, 0) and At(*, 0), At(0, *) must all be zero.
    let img = constant(8, 6, 200);
    let ii = IntegralImage::from_gray(&img);
    assert_eq!(ii.at(0, 0), 0);
    for x in 0..=8 {
        assert_eq!(ii.at(x, 0), 0);
    }
    for y in 0..=6 {
        assert_eq!(ii.at(0, y), 0);
    }
}

#[test]
fn integral_at_corner_equals_total_pixel_sum() {
    // II[W, H] (the bottom-right corner of the padded table) must equal
    // the total of all pixels.
    let mut data = Vec::with_capacity(4 * 4);
    for y in 0..4 {
        for x in 0..4 {
            data.push(((x + y * 4) * 13 + 5) as u8);
        }
    }
    let img = GrayImage::from_vec(data, 4, 4);
    let ii = IntegralImage::from_gray(&img);
    let total: u64 = img.as_slice().iter().map(|&v| v as u64).sum();
    assert_eq!(ii.at(4, 4), total);
}
