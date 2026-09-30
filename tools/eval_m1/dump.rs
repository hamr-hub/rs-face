//! Per-image pipeline dump: decode → detect → align → embed, with PNGs on disk.
//!
//! Answers "where did this image lose its identity" by writing out every
//! intermediate product the recogniser actually consumed, rather than only
//! reporting a similarity number:
//!
//!   `<name>.decoded.png`  — the pixels after our hand-rolled JPEG decode
//!   `<name>.cropNN.png`   — the 112x112 ArcFace-aligned crop, per detection
//!
//! A wrong-looking `.decoded.png` indicts the decoder; a good decode with a
//! broken-looking crop indicts detection/alignment; good crops with a wrong
//! embedding indicts the model wiring.
//!
//! Usage:
//!   ORT_DYLIB_PATH=data/eval_m1/libonnxruntime.so \
//!     cargo run --release --manifest-path tools/eval_m1/Cargo.toml --bin dump -- \
//!     --out data/eval_diag/run1 <image>...

use rsface::arcface_recognizer::ArcFaceRecognizer;
use rsface::face::FaceDetection;
use rsface::image::jpeg::decode_jpeg_rgb;
use rsface::image::png;
use rsface::image::RgbImage;
use rsface::models;
use rsface::onnx::SessionConfig;
use rsface::scrfd::ScrfdConfig;
use rsface::scrfd_detector::ScrfdDetector;
use std::path::PathBuf;

fn stats(name: &str, img: &RgbImage) {
    let s = img.as_slice();
    let n = s.len().max(1);
    let mean = s.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var = s.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    println!(
        "  {name}: {}x{} mean={mean:.1} std={:.1} min={} max={}",
        img.width(),
        img.height(),
        var.sqrt(),
        s.iter().min().unwrap_or(&0),
        s.iter().max().unwrap_or(&0),
    );
}

/// Residual between detected landmarks warped through the similarity transform and the
/// canonical ArcFace layout — a sub-pixel residual means the detector agreed with the
/// training layout, large residual means the crop is computed off-centre and every
/// embedding will suffer. Printed in the dump so a wrong alignment is visible without
/// staring at the crop PNG.
fn alignment_residual(lms: &rsface::face::Landmarks) -> f32 {
    use rsface::align::estimate_arcface_transform;
    let Some(t) = estimate_arcface_transform(lms) else {
        return f32::NAN;
    };
    let mut worst = 0.0f32;
    for i in 0..rsface::face::NUM_LANDMARKS {
        let (x, y) = t.apply(lms.points[i].0, lms.points[i].1);
        let (rx, ry) = rsface::align::ARCFACE_REFERENCE_LANDMARKS[i];
        worst = worst.max(((x - rx).powi(2) + (y - ry).powi(2)).sqrt());
    }
    worst
}

fn write_png(path: &PathBuf, img: &RgbImage) {
    if let Ok(f) = std::fs::File::create(path) {
        let mut w = std::io::BufWriter::new(f);
        let _ = png::write_png_rgb(&mut w, img);
    }
}

fn stem(p: &PathBuf) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "img".into())
}

fn main() {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut out_dir = PathBuf::from("data/eval_diag/dump");
    let mut models_dir = PathBuf::from("models");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out_dir = PathBuf::from(args.next().expect("<DIR>")),
            "--models-dir" => models_dir = PathBuf::from(args.next().expect("<DIR>")),
            other => files.push(PathBuf::from(other)),
        }
    }
    if files.is_empty() {
        eprintln!("usage: dump [--out DIR] <image>...");
        std::process::exit(2);
    }

    let det_spec = models::REGISTRY
        .iter()
        .find(|s| s.file_name == "det_10g.onnx")
        .expect("det_10g spec");
    let rec_spec = models::REGISTRY
        .iter()
        .find(|s| s.file_name == "w600k_r50.onnx")
        .expect("w600k spec");
    let cfg = SessionConfig::default();
    let detector = ScrfdDetector::open(
        &models_dir.join("det_10g.onnx"),
        Some(det_spec),
        &cfg,
        ScrfdConfig::default(),
    )
    .expect("SCRFD");
    let recog = ArcFaceRecognizer::open(
        &models_dir.join("w600k_r50.onnx"),
        Some(rec_spec),
        &cfg,
    )
    .expect("ArcFace");

    let _ = std::fs::create_dir_all(&out_dir);

    for f in &files {
        println!("\n=== {} ===", f.display());
        let bytes = match std::fs::read(f) {
            Ok(b) => b,
            Err(e) => {
                println!("  READ FAIL: {e}");
                continue;
            }
        };
        let img = match decode_jpeg_rgb(&bytes) {
            Ok(i) => i,
            Err(e) => {
                println!("  DECODE FAIL: {e}");
                continue;
            }
        };
        stats("decoded", &img);
        let s = stem(f);
        write_png(&out_dir.join(format!("{s}.decoded.png")), &img);

        let mut dets: Vec<FaceDetection> = detector.detect_rgb(&img).expect("detect");
        dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        println!("  detections: {}", dets.len());
        for (i, d) in dets.iter().take(3).enumerate() {
            let (x, y, w, h) = (d.x1, d.y1, d.width(), d.height());
            println!(
                "    #{i} score={:.4} box=({x:.1},{y:.1},{w:.1},{h:.1}) lms={}",
                d.score,
                d.landmarks.is_some()
            );
            if let Some(lms) = d.landmarks {
                let res = alignment_residual(&lms);
                println!("        residual_to_arcface_canonical={res:.2}px");
                for (k, p) in lms.points.iter().enumerate() {
                    println!("        p{k} = ({:.1}, {:.1})", p.0, p.1);
                }
                if let Some(crop) = rsface::align::norm_crop(&img, &lms) {
                    stats(&format!("crop#{i}"), &crop);
                    write_png(&out_dir.join(format!("{s}.crop{i}.png")), &crop);
                    match recog.embed(&img, &lms) {
                        Ok(e) => {
                            let v = e.as_slice();
                            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                            println!(
                                "    embed#{i} dim={} norm={n:.4} head={:.4},{:.4},{:.4}",
                                v.len(),
                                v[0],
                                v[1],
                                v[2]
                            );
                        }
                        Err(err) => println!("    embed#{i} FAIL: {err}"),
                    }
                    match recog.embed_tta(&img, &lms) {
                        Ok(e) => {
                            let v = e.as_slice();
                            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                            println!(
                                "    tta#{i} dim={} norm={n:.4} head={:.4},{:.4},{:.4}",
                                v.len(),
                                v[0],
                                v[1],
                                v[2]
                            );
                        }
                        Err(err) => println!("    tta#{i} FAIL: {err}"),
                    }
                } else {
                    println!("    crop#{i} FAILED to align (degenerate landmarks)");
                }
            }
        }
    }
}
