//! Layer-by-layer diagnostic for the 0-recall report.
//!
//! Isolates the three places the signal can die before a box exists:
//!   1. JPEG decode      — is the decoded pixel buffer a plausible image?
//!   2. preprocessing    — is the NCHW tensor the model expects?
//!   3. model output     — what raw scores does the graph actually emit?
//!
//! Prints the *raw* max score per stride so a threshold can never be the
//! suspect by assumption, and writes a PNG of the decoded frame so a
//! human can look at what the model was handed.

use rsface::image::jpeg::decode_jpeg_rgb;
use rsface::image::RgbImage;
use rsface::models;
use rsface::onnx::{open_session, SessionConfig};
use rsface::scrfd::{self, ScrfdConfig, INPUT_MEAN, INPUT_STD, STRIDES};
use std::path::PathBuf;

fn stats(name: &str, img: &RgbImage) {
    let s = img.as_slice();
    let n = s.len();
    let mean = s.iter().map(|&v| v as f64).sum::<f64>() / n.max(1) as f64;
    let var = s
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n.max(1) as f64;
    println!(
        "  {name}: {}x{} | mean={mean:.1} std={:.1} min={} max={}",
        img.width(),
        img.height(),
        var.sqrt(),
        s.iter().min().unwrap_or(&0),
        s.iter().max().unwrap_or(&0),
    );
}

fn main() {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut dump_dir = PathBuf::from("data/eval_diag");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dump-dir" => dump_dir = PathBuf::from(args.next().expect("<DIR>")),
            other => files.push(PathBuf::from(other)),
        }
    }
    if files.is_empty() {
        files.push(PathBuf::from("data/eval_m1/split/gallery/000_George_W_Bush_0315.jpg"));
        files.push(PathBuf::from("platform/testdata/biden.jpg"));
        files.push(PathBuf::from("platform/testdata/lena.jpg"));
    }

    let det_spec = models::REGISTRY
        .iter()
        .find(|s| s.file_name == "det_10g.onnx")
        .expect("det_10g spec");
    let size = det_spec.input_size;
    let session_cfg =
        SessionConfig::default().with_input_shape(&[1, 3, size, size]);
    let session = open_session(
        PathBuf::from("models/det_10g.onnx").as_path(),
        Some(det_spec),
        &session_cfg,
    )
    .expect("open session");

    let _ = std::fs::create_dir_all(&dump_dir);

    for f in &files {
        println!("\n=== {} ===", f.display());
        let bytes = match std::fs::read(f) {
            Ok(b) => b,
            Err(e) => {
                println!("  READ FAIL: {e}");
                continue;
            }
        };
        println!("  file bytes: {}", bytes.len());
        let img = match decode_jpeg_rgb(&bytes) {
            Ok(i) => i,
            Err(e) => {
                println!("  DECODE FAIL: {e}");
                continue;
            }
        };
        stats("decoded", &img);

        // Dump the decoded frame as PNG so it can be eyeballed directly.
        let stem = f
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("frame");
        let png_path = dump_dir.join(format!("{stem}.png"));
        if let Ok(mut f_out) = std::fs::File::create(&png_path) {
            if rsface::image::png::write_png_rgb(&mut f_out, &img).is_ok() {
                println!("  dumped decoded frame -> {}", png_path.display());
            }
        }

        // Preprocess and sanity-check the tensor itself.
        let (input, lb) = scrfd::preprocess(&img, size);
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &v in &input {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let expect_lo = (0.0 - INPUT_MEAN) / INPUT_STD;
        let expect_hi = (255.0 - INPUT_MEAN) / INPUT_STD;
        println!(
            "  tensor: len={} (want {}) range=[{lo:.3},{hi:.3}] (want ~[{expect_lo:.3},{expect_hi:.3}])",
            input.len(),
            3 * size * size
        );
        println!("  letterbox: scale={:.4} resized={}x{}", lb.scale, lb.resized_w, lb.resized_h);

        // Raw forward pass; report the actual score distribution per stride.
        let outputs = session.run(&input, &[1, 3, size, size]).expect("run");
        println!("  outputs: {} tensors", outputs.len());
        for (i, t) in outputs.iter().enumerate() {
            let n = t.data.len();
            let max = t.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = t.data.iter().cloned().fold(f32::INFINITY, f32::min);
            let over = t.data.iter().filter(|&&v| v > 0.5).count();
            println!(
                "    [{i}] shape={:?} n={n} min={min:.4} max={max:.4} >0.5:{over}",
                t.shape
            );
        }

        // Now the full detector, at a deliberately low threshold.
        let det = rsface::scrfd_detector::ScrfdDetector::open(
            PathBuf::from("models/det_10g.onnx").as_path(),
            Some(det_spec),
            &SessionConfig::default(),
            ScrfdConfig::default().with_score_threshold(0.05),
        )
        .expect("open detector");
        let dets = det.detect_rgb(&img).expect("detect");
        println!("  detector @thr=0.05 -> {} faces", dets.len());
        for d in dets.iter().take(5) {
            println!(
                "    score={:.4} box=({:.1},{:.1})-({:.1},{:.1}) lm={}",
                d.score,
                d.x1,
                d.y1,
                d.x2,
                d.y2,
                d.landmarks.is_some()
            );
        }
        let _ = STRIDES.len();
    }
}
