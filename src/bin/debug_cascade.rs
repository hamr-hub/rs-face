//! Debug helper: load a cascade, run it on a single image, and print
//! per-stage decisions for a few candidate windows. Use to verify the
//! cascade and feature evaluation are correct.

use rsface::haar::{Cascade, EvalCache};
use rsface::image::{codec, GrayImage, RgbImage};
use rsface::integral::{IntegralImage, RotatedIntegralImage};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: debug_cascade <cascade.rfcf> <image.ppm>");
        std::process::exit(2);
    }
    let cascade = Cascade::load(Path::new(&args[1])).expect("load cascade");
    let mut f = std::fs::File::open(&args[2]).expect("open image");
    let rgb = codec::read_ppm(&mut f).expect("decode ppm");
    let gray = rgb.to_gray();
    eprintln!("image: {}x{}", gray.width(), gray.height());

    let ii = IntegralImage::from_gray(&gray);
    let ri = RotatedIntegralImage::from_gray(&gray);
    let ww = cascade.window_w;
    let wh = cascade.window_h;
    eprintln!("window: {}x{}", ww, wh);
    let mut cache = EvalCache::new(cascade.num_features());

    // Probe many windows across the image to find faces.
    let step = 32;
    let mut probes = Vec::new();
    let mut y = 0;
    while y + wh <= gray.height() {
        let mut x = 0;
        while x + ww <= gray.width() {
            probes.push((x, y));
            x += step;
        }
        y += step;
    }
    eprintln!("scanning {} windows", probes.len());

    // Try evaluating at the exact cascade-scale position: face 101x101 at (338,89)
    // means cascade scale = 101/24 = 4.21. At this scale, image is 480/4.21 × 854/4.21
    // and the face is at (338/4.21, 89/4.21) = (80, 21) in 24x24 size.
    for scale in [4.21] {
        let nw = ((gray.width() as f32) / scale).round() as usize;
        let nh = ((gray.height() as f32) / scale).round() as usize;
        let scaled = gray.resize_area(nw, nh);
        // Save scaled image to compare with Python.
        let mut rgb = RgbImage::new(nw, nh);
        for y in 0..nh {
            for x in 0..nw {
                let v = scaled[(x, y)];
                let row = rgb.row_mut(y);
                row[x * 3] = v;
                row[x * 3 + 1] = v;
                row[x * 3 + 2] = v;
            }
        }
        let _ = std::fs::File::create("/tmp/rsface_test/scaled_rust.ppm").map(|mut f| {
            use std::io::Write;
            write!(f, "P6\n{} {}\n255\n", nw, nh).unwrap();
            f.write_all(rgb.as_slice()).unwrap();
        });
        let sii = IntegralImage::from_gray(&scaled);
        let sri = RotatedIntegralImage::from_gray(&scaled);
        let fx = ((338.0 / scale).round() as usize).min(nw - ww);
        let fy = ((89.0 / scale).round() as usize).min(nh - wh);
        eprintln!(
            "\n=== scale {:.2} → image {}x{}, face at ({},{}) ===",
            scale, nw, nh, fx, fy
        );
        eprintln!("scaled image saved to /tmp/rsface_test/scaled_rust.ppm");
        // Direct rect sums for feature 1
        eprintln!(
            "rect 0 sum (image 86,25-98,32) = {}",
            sii.rect_sum(86, 25, 98, 32)
        );
        eprintln!(
            "rect 1 sum (image 90,25-94,32) = {}",
            sii.rect_sum(90, 25, 94, 32)
        );
        // Per-feature detail
        let f0 = &cascade.features_debug()[0];
        let r0 = f0.eval(&sii, &sri, fx, fy, ww, wh, nw, nh);
        eprintln!("feature 0 response = {} (expected: -0.953)", r0);
        // All features in stage 0
        for (i, w) in cascade.stages_debug()[0].weak_features.iter().enumerate() {
            let f = &cascade.features_debug()[w.feature_index as usize];
            let r = f.eval(&sii, &sri, fx, fy, ww, wh, nw, nh);
            eprintln!(
                "stage 0 weak {}: feat {} r={} thr={} → {}",
                i,
                w.feature_index,
                r,
                w.threshold,
                if r > w.threshold {
                    format!("right={}", w.right_val)
                } else {
                    format!("left={}", w.left_val)
                }
            );
        }
        let r = cascade.classify(&sii, &sri, fx, fy, &mut cache);
        eprintln!(
            "classify: {}",
            match r {
                Some(s) => format!("PASS score={:.4}", s),
                None => "REJECT".to_string(),
            }
        );
        // Sum of stage 0 weak values
        let mut stage0_sum = 0.0f32;
        for w in &cascade.stages_debug()[0].weak_features {
            let f = &cascade.features_debug()[w.feature_index as usize];
            let r = f.eval(&sii, &sri, fx, fy, ww, wh, nw, nh);
            let v = if r > w.threshold {
                w.right_val
            } else {
                w.left_val
            };
            stage0_sum += v;
        }
        eprintln!(
            "stage 0 manual sum = {} (threshold {})",
            stage0_sum,
            cascade.stages_debug()[0].stage_threshold
        );

        // Run all stages manually
        for stage_idx in 0..cascade.num_stages() {
            let mut stage_sum = 0.0f32;
            for w in &cascade.stages_debug()[stage_idx].weak_features {
                let f = &cascade.features_debug()[w.feature_index as usize];
                let r = f.eval(&sii, &sri, fx, fy, ww, wh, nw, nh);
                let v = if r > w.threshold {
                    w.right_val
                } else {
                    w.left_val
                };
                stage_sum += v;
            }
            let pass = stage_sum >= cascade.stages_debug()[stage_idx].stage_threshold;
            eprintln!(
                "manual stage {:2} sum={:8.4} threshold={:8.4} {}",
                stage_idx,
                stage_sum,
                cascade.stages_debug()[stage_idx].stage_threshold,
                if pass { "PASS" } else { "REJECT" }
            );
            if !pass {
                break;
            }
        }
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut best = (0usize, 0usize, f32::NEG_INFINITY);
    for (x, y) in &probes {
        match cascade.classify(&ii, &ri, *x, *y, &mut cache) {
            Some(score) => {
                if score > best_score {
                    best_score = score;
                    best = (*x, *y, score);
                }
            }
            None => {}
        }
    }
    eprintln!("best window: ({}, {}) score={:.4}", best.0, best.1, best.2);
    eprintln!("\n=== detail at best window ({}, {}) ===", best.0, best.1);
    let (x, y) = (best.0, best.1);
    for stage_idx in 0..cascade.num_stages() {
        if let Some((sum, _details)) = cascade.eval_stage(&ii, &ri, x, y, stage_idx) {
            let pass = sum >= cascade.stages_debug()[stage_idx].stage_threshold;
            eprintln!(
                "  stage {:2} sum={:8.4} threshold={:8.4} → {}",
                stage_idx,
                sum,
                cascade.stages_debug()[stage_idx].stage_threshold,
                if pass { "PASS" } else { "REJECT" }
            );
            if !pass {
                break;
            }
        }
    }

    // Also probe the windows where OpenCV detected faces.
    eprintln!("\n=== opencv-detected face positions ===");
    for &(fx, fy, fw, fh) in &[(338usize, 89, 101, 101), (181, 110, 187, 187)] {
        // Use the top-left of the detection as our window.
        let cx = fx + fw / 2;
        let cy = fy + fh / 2;
        eprintln!(
            "\n  face {}x{} at ({},{}) center=({},{})",
            fw, fh, fx, fy, cx, cy
        );
        // Scan a 7x7 grid around the face center.
        for dy in [-20i32, -10, 0, 10, 20] {
            for dx in [-20i32, -10, 0, 10, 20] {
                let x = ((cx as i32) + dx - ww as i32 / 2).max(0) as usize;
                let y = ((cy as i32) + dy - wh as i32 / 2).max(0) as usize;
                if x + ww > gray.width() || y + wh > gray.height() {
                    continue;
                }
                match cascade.classify(&ii, &ri, x, y, &mut cache) {
                    Some(score) => eprintln!("    ({},{}) PASS score={:.4}", x, y, score),
                    None => {}
                }
            }
        }
    }
    for (x, y) in probes {
        let x = x.saturating_sub(ww / 2);
        let y = y.saturating_sub(wh / 2);
        eprintln!("\n=== window at ({}, {}) ===", x, y);
        for stage_idx in 0..cascade.num_stages() {
            if let Some((sum, _details)) = cascade.eval_stage(&ii, &ri, x, y, stage_idx) {
                let pass = sum >= cascade.stages_debug()[stage_idx].stage_threshold;
                eprintln!(
                    "  stage {:2} sum={:8.4} threshold={:8.4} → {}",
                    stage_idx,
                    sum,
                    cascade.stages_debug()[stage_idx].stage_threshold,
                    if pass { "PASS" } else { "REJECT" }
                );
                if !pass {
                    break;
                }
            }
        }
        // Final classify
        match cascade.classify(&ii, &ri, x, y, &mut cache) {
            Some(score) => eprintln!("  FINAL: PASS score={:.4}", score),
            None => eprintln!("  FINAL: REJECT"),
        }
    }
}
