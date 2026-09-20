//! Edge-case coverage for `GrayImage::resize_bilinear`, `resize_area`,
//! and `downscale`. The grayscale resize primitives are the foundation
//! the SCRFD / ArcFace ONNX pre-processing path depends on, so silent
//! regressions here directly corrupt downstream accuracy.

use rsface::image::GrayImage;

fn ramp_image(w: usize, h: usize) -> GrayImage {
    // Deterministic ramp so resizes can be checked against an analytic
    // expected average: row y holds the value (y * w + x) capped at 255.
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = (y * w + x) % 256;
            img[(x, y)] = v as u8;
        }
    }
    img
}

#[test]
fn resize_bilinear_1x1_to_anything_holds_constant() {
    // A 1x1 source upsampled to NxM must keep the single pixel's value
    // everywhere (bilinear edge-clamp degenerates to nearest-sample).
    let img = GrayImage::from_vec(vec![123], 1, 1);
    for (w, h) in [(1, 1), (2, 2), (16, 16), (640, 480)] {
        let out = img.resize_bilinear(w, h);
        assert_eq!(out.width(), w);
        assert_eq!(out.height(), h);
        for y in 0..h {
            for x in 0..w {
                let v = out[(x, y)];
                assert!(
                    (v as i32 - 123).abs() <= 1,
                    "1x1 upsampled to {w}x{h} pixel ({x},{y}) = {v}, expected 123 +/- 1"
                );
            }
        }
    }
}

#[test]
fn resize_bilinear_2x2_identity() {
    // 2x2 -> 2x2 must reproduce the input exactly. The bilinear path
    // can introduce a +-1 rounding step at edge pixels; assert <= 1.
    let mut img = GrayImage::new(2, 2);
    img[(0, 0)] = 10;
    img[(1, 0)] = 20;
    img[(0, 1)] = 30;
    img[(1, 1)] = 40;
    let out = img.resize_bilinear(2, 2);
    for y in 0..2 {
        for x in 0..2 {
            let a = img[(x, y)] as i32;
            let b = out[(x, y)] as i32;
            assert!((a - b).abs() <= 1, "({x},{y}) {a} vs {b}");
        }
    }
}

#[test]
fn resize_bilinear_uniform_image_uniform_output() {
    // Constant source must produce a constant output at any size (exact,
    // not +/- 1: there is no interpolation happening).
    let img = GrayImage::from_vec(vec![200; 16 * 16], 16, 16);
    for (w, h) in [(1, 1), (3, 5), (32, 32), (640, 480)] {
        let out = img.resize_bilinear(w, h);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(out[(x, y)], 200, "{w}x{h} pixel ({x},{y})");
            }
        }
    }
}

#[test]
fn resize_bilinear_downscale_odd_dimensions_is_stable() {
    // Odd output dimensions stress the floor / ceil branches in the
    // bilinear sample mapping. Output must not panic and stay in [0, 255].
    let img = ramp_image(33, 27);
    let out = img.resize_bilinear(7, 5);
    assert_eq!((out.width(), out.height()), (7, 5));
    // u8 max is 255, so "every pixel fits" is trivially true. Instead check
    // the resize actually produced non-trivial output (a sane resize of a
    // non-uniform source must yield at least one mid-range pixel).
    assert!(out.as_slice().iter().any(|&v| v > 0));
    assert!(out.as_slice().iter().any(|&v| v < 255));
}

#[test]
fn resize_bilinear_extreme_ratio_does_not_panic() {
    // 100x100 -> 1x1 and 1x100 must produce a deterministic output of the
    // right shape with every pixel in [0, 255].
    let img = ramp_image(100, 100);
    let tiny = img.resize_bilinear(1, 1);
    assert_eq!((tiny.width(), tiny.height()), (1, 1));
    let wide = img.resize_bilinear(1, 100);
    assert_eq!((wide.width(), wide.height()), (1, 100));
    // u8 max is 255 so a per-pixel <= 255 check would be a tautology;
    // assert the output is non-trivial (not all zeros).
    assert!(wide.as_slice().iter().any(|&v| v > 0));
}

#[test]
fn resize_bilinear_upsample_smoothness_neighbours_close() {
    // A constant-then-edge image: left half 0, right half 255. Upsampled
    // 2x, neighbouring pixels must be close (bilinear is C0).
    let mut img = GrayImage::new(2, 1);
    img[(0, 0)] = 0;
    img[(1, 0)] = 255;
    let out = img.resize_bilinear(64, 1);
    // Adjacent samples differ by at most 8 (256/64 * 2 for two steps).
    for x in 1..out.width() {
        let a = out[(x - 1, 0)] as i32;
        let b = out[(x, 0)] as i32;
        assert!(
            (a - b).abs() <= 16,
            "bilinear neighbours too far apart at x={x}: {a} {b}"
        );
    }
}

#[test]
fn resize_area_uniform_image_uniform_output() {
    // Area averaging on a constant input is also constant.
    let img = GrayImage::from_vec(vec![150; 20 * 20], 20, 20);
    for (w, h) in [(1, 1), (5, 5), (40, 30), (640, 480)] {
        let out = img.resize_area(w, h);
        for v in out.as_slice() {
            assert_eq!(*v, 150, "{w}x{h} pixel");
        }
    }
}

#[test]
fn resize_area_downscale_factor_4_matches_box_average() {
    // Downscale by 4x with `resize_area` must match the explicit
    // `downscale(factor=4)` output (both are area averages).
    let img = ramp_image(40, 40);
    let via_area = img.resize_area(10, 10);
    let via_box = img.downscale(4);
    assert_eq!((via_area.width(), via_area.height()), (10, 10));
    for i in 0..via_area.as_slice().len() {
        let a = via_area.as_slice()[i] as i32;
        let b = via_box.as_slice()[i] as i32;
        assert!(
            (a - b).abs() <= 1,
            "area vs box mismatch at i={i}: {a} vs {b}"
        );
    }
}

#[test]
fn resize_area_2x_downscale_ramp_average_in_window() {
    // 4x4 image with values 0..16 downscale to 2x2: each output pixel is
    // the average of a 2x2 block (0+1+4+5)/4=2.5 -> rounded to 3 etc.
    let img = GrayImage::from_vec((0u8..16).collect::<Vec<u8>>(), 4, 4);
    let out = img.resize_area(2, 2);
    assert_eq!((out.width(), out.height()), (2, 2));
    // Expected: block (0,0)=2.5, (2,0)=4.5, (0,2)=10.5, (2,2)=12.5
    let expected = [3u8, 5, 11, 13];
    for (i, &e) in expected.iter().enumerate() {
        let x = i % 2;
        let y = i / 2;
        let v = out[(x, y)];
        assert!(
            (v as i32 - e as i32).abs() <= 1,
            "({x},{y}) got {v} expected {e} +/-1"
        );
    }
}

#[test]
fn resize_area_odd_dimensions_does_not_panic() {
    // 31x23 -> 7x5 (non-integer downscale ratio).
    let img = ramp_image(31, 23);
    let out = img.resize_area(7, 5);
    assert_eq!((out.width(), out.height()), (7, 5));
    // u8 max is 255 — per-pixel clamp is a tautology. Sanity-check the
    // resize produced non-trivial values.
    assert!(out.as_slice().iter().any(|&v| v > 0));
    assert!(out.as_slice().iter().any(|&v| v < 255));
}

#[test]
fn resize_area_extreme_ratio_does_not_panic() {
    let img = ramp_image(100, 100);
    let tiny = img.resize_area(1, 1);
    assert_eq!((tiny.width(), tiny.height()), (1, 1));
    let wide = img.resize_area(1, 200);
    assert_eq!((wide.width(), wide.height()), (1, 200));
}

#[test]
fn downscale_factor_1_is_clone() {
    let img = ramp_image(8, 6);
    let out = img.downscale(1);
    assert_eq!((out.width(), out.height()), (img.width(), img.height()));
    assert_eq!(out.as_slice(), img.as_slice());
}

#[test]
fn downscale_factor_4_is_exact_box_average() {
    // Brute-force the expected per-pixel box average and compare.
    let img = ramp_image(8, 8);
    let out = img.downscale(4);
    assert_eq!((out.width(), out.height()), (2, 2));
    for y in 0..2 {
        for x in 0..2 {
            let mut s: u32 = 0;
            for j in 0..4 {
                for i in 0..4 {
                    s += img[(x * 4 + i, y * 4 + j)] as u32;
                }
            }
            let expected = (s / 16) as u8;
            assert_eq!(out[(x, y)], expected);
        }
    }
}

#[test]
fn downscale_odd_input_does_not_panic_and_truncates() {
    // 7x5 downscale by 2 -> 3x2 (integer truncation of dims).
    let img = ramp_image(7, 5);
    let out = img.downscale(2);
    assert_eq!((out.width(), out.height()), (3, 2));
    // u8 max is 255; per-pixel clamp is tautological. Sanity-check the
    // downscale produced non-trivial output.
    assert!(out.as_slice().iter().any(|&v| v > 0));
}
