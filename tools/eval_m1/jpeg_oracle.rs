//! Pixel-accuracy grader for the rs-face JPEG decoder.
//!
//! Grades every case in data/eval_jpeg/manifest.json against the PIL/libjpeg
//! reference `.raw` and prints a per-case table plus a pass/fail verdict.
//!
//! Byte equality is the wrong bar: our decoder uses a float IDCT where libjpeg
//! uses a fixed-point one, and a different YCbCr->RGB rounding, so an exact
//! match is neither expected nor necessary. The bar that *does* matter for
//! face detection is PSNR — below roughly 35 dB the decode is visually broken
//! and a CNN trained on clean pixels degrades badly.
//!
//! Usage:
//!   cargo run --release --manifest-path tools/eval_m1/Cargo.toml --bin jpeg_oracle
//!   ... -- --verbose          print every case, not just failures
//!   ... -- --psnr 35          tighten the pass threshold

use rsface::image::jpeg::decode_jpeg_rgb;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Case {
    name: String,
    file: String,
    raw: String,
    width: usize,
    height: usize,
}

struct Row {
    name: String,
    mae: f64,
    psnr: f64,
    max_err: u8,
    decoded: bool,
    err: String,
}

fn grade(dir: &PathBuf, c: &Case, dump_raw: Option<&PathBuf>) -> Row {
    let jpath = dir.join(&c.file);
    let bytes = match std::fs::read(&jpath) {
        Ok(b) => b,
        Err(e) => {
            return Row {
                name: c.name.clone(),
                mae: f64::INFINITY,
                psnr: f64::NEG_INFINITY,
                max_err: 255,
                decoded: false,
                err: format!("read: {e}"),
            }
        }
    };
    let img = match decode_jpeg_rgb(&bytes) {
        Ok(i) => i,
        Err(e) => {
            return Row {
                name: c.name.clone(),
                mae: f64::INFINITY,
                psnr: f64::NEG_INFINITY,
                max_err: 255,
                decoded: false,
                err: format!("decode: {e}"),
            }
        }
    };
    if img.width() != c.width || img.height() != c.height {
        return Row {
            name: c.name.clone(),
            mae: f64::INFINITY,
            psnr: f64::NEG_INFINITY,
            max_err: 255,
            decoded: true,
            err: format!("size {}x{} want {}x{}", img.width(), img.height(), c.width, c.height),
        };
    }
    let refbytes = match std::fs::read(dir.join(&c.raw)) {
        Ok(b) => b,
        Err(e) => {
            return Row {
                name: c.name.clone(),
                mae: f64::INFINITY,
                psnr: f64::NEG_INFINITY,
                max_err: 255,
                decoded: true,
                err: format!("ref: {e}"),
            }
        }
    };
    let got = img.as_slice();
    if let Some(d) = dump_raw {
        let _ = std::fs::create_dir_all(d);
        let _ = std::fs::write(d.join(format!("{}.our.raw", c.name)), got);
    }
    if got.len() != refbytes.len() {
        return Row {
            name: c.name.clone(),
            mae: f64::INFINITY,
            psnr: f64::NEG_INFINITY,
            max_err: 255,
            decoded: true,
            err: format!("len {} want {}", got.len(), refbytes.len()),
        };
    }

    let n = got.len() as f64;
    let mut se = 0.0f64;
    let mut ae = 0.0f64;
    let mut max_err = 0u8;
    for i in 0..got.len() {
        let d = (got[i] as f64 - refbytes[i] as f64).abs();
        se += d * d;
        ae += d;
        if d > max_err as f64 {
            max_err = got[i].abs_diff(refbytes[i]);
        }
    }
    let mse = se / n;
    let mae = ae / n;
    // PSNR with peak 255. mse==0 (exact) -> report a large finite number
    // rather than infinity so sorting and formatting stay well-behaved.
    let psnr = if mse <= f64::MIN_POSITIVE { 200.0 } else { 10.0 * (255.0f64 * 255.0 / mse).log10() };
    Row {
        name: c.name.clone(),
        mae,
        psnr,
        max_err,
        decoded: true,
        err: String::new(),
    }
}

fn main() {
    let mut dir = PathBuf::from("data/eval_jpeg");
    let mut verbose = false;
    let mut threshold = 35.0f64;
    let mut dump_raw: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = PathBuf::from(args.next().expect("<DIR>")),
            "--verbose" | "-v" => verbose = true,
            "--psnr" => threshold = args.next().expect("<F64>").parse().expect("f64"),
            "--dump-raw" => dump_raw = Some(PathBuf::from(args.next().expect("<DIR>"))),
            other => {
                eprintln!("unknown arg {other}");
                return;
            }
        }
    }

    let manifest: Vec<Case> = serde_json::from_slice(
        &std::fs::read(dir.join("manifest.json")).expect("read manifest"),
    )
    .expect("parse manifest");

    let mut rows: Vec<Row> = manifest
        .iter()
        .map(|c| grade(&dir, c, dump_raw.as_ref()))
        .collect();
    rows.sort_by(|a, b| a.psnr.partial_cmp(&b.psnr).unwrap());

    println!("{:<34} {:>7} {:>8} {:>6}  {}", "case", "MAE", "PSNR", "maxErr", "status");
    println!("{}", "-".repeat(78));
    let mut pass = 0usize;
    let mut fail = 0usize;
    for r in &rows {
        let ok = r.psnr >= threshold;
        if ok {
            pass += 1;
        } else {
            fail += 1;
        }
        if !ok || verbose {
            println!(
                "{:<34} {:>7.3} {:>8.2} {:>6}  {}",
                r.name,
                r.mae,
                r.psnr,
                r.max_err,
                if ok { "ok" } else { "FAIL" }
            );
            if !r.err.is_empty() {
                println!("    ^ {}", r.err);
            }
        }
    }
    println!("{}", "-".repeat(78));
    println!(
        "{}/{} pass at PSNR >= {:.1} dB   (worst: {} @ {:.2} dB)",
        pass,
        rows.len(),
        threshold,
        rows.first().map(|r| r.name.as_str()).unwrap_or("-"),
        rows.first().map(|r| r.psnr).unwrap_or(0.0),
    );
    if fail > 0 {
        std::process::exit(1);
    }
}
