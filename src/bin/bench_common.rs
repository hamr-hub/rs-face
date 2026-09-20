//! Shared helpers for the zero-dependency recogniser accuracy benchmarks.
//!
//! Included with `#[path = "bench_common.rs"] mod bench_common;` from `bench_lbph`
//! and `bench_eigenface`; it is deliberately NOT a bin target itself (Cargo.toml
//! sets `autobins = false`), so cargo neither builds nor warns about it standalone.

use std::collections::BTreeMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;

use rsface::image::codec;
use rsface::image::png;
use rsface::image::{GrayImage, RgbImage};

/// One loaded crop with its ground-truth label.
pub struct Crop {
    pub label: String,
    pub img: GrayImage,
}

/// Recursively load `*.pgm` / `*.ppm` / `*.png` crops under `root`.
///
/// Ground-truth labels come from the filename convention
/// `<identity>__<source>__<frame>.*` (part before the first `__`), falling back to
/// the parent directory name.
///
/// Entries are visited in sorted path order: `read_dir` alone returns
/// filesystem-defined order, and exact distance ties in the LOO loop would make
/// the reported rank-1 vary between runs.
pub fn load_crops(root: &Path) -> Vec<Crop> {
    let mut out = Vec::new();
    walk(root, &mut out);
    out
}

fn walk(dir: &Path, out: &mut Vec<Crop>) {
    let mut paths: Vec<_> = fs::read_dir(dir)
        .expect("read crops dir")
        .map(|entry| entry.expect("dir entry").path())
        .collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            walk(&path, out);
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        let img: Option<GrayImage> = match ext.to_ascii_lowercase().as_str() {
            "pgm" => {
                let f = fs::File::open(&path).expect("open pgm");
                codec::read_pgm(&mut BufReader::new(f)).ok()
            }
            "ppm" => codec::read_ppm(&mut BufReader::new(
                fs::File::open(&path).expect("open ppm"),
            ))
            .ok()
            .map(|rgb: RgbImage| rgb.to_gray()),
            "png" => png::decode_to_gray(&mut BufReader::new(
                fs::File::open(&path).expect("open png"),
            ))
            .ok(),
            _ => None,
        };
        let Some(img) = img else { continue };
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let label = match stem.split_once("__") {
            Some((label, _)) => label.to_string(),
            None => path
                .parent()
                .and_then(|p| p.file_name())
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or(stem),
        };
        out.push(Crop { label, img });
    }
}

/// Per-label crop counts.
pub fn label_counts(crops: &[Crop]) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for c in crops {
        *counts.entry(c.label.clone()).or_default() += 1;
    }
    counts
}

/// Accuracy of the decision "same iff distance <= t" over both pair populations.
pub fn pair_accuracy(t: f32, same: &[f32], different: &[f32]) -> f32 {
    let tp = same.iter().filter(|&&d| d <= t).count();
    let tn = different.iter().filter(|&&d| d > t).count();
    (tp + tn) as f32 / (same.len() + different.len()) as f32
}

/// Threshold where FAR ≈ FRR (equal error rate operating point).
pub fn eer(same: &[f32], different: &[f32]) -> f32 {
    let mut candidates: Vec<f32> = same.iter().chain(different.iter()).copied().collect();
    candidates.sort_by(f32::total_cmp);
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let mut best_t = 0.0;
    let mut best_gap = f32::MAX;
    for w in candidates.windows(2) {
        let t = (w[0] + w[1]) / 2.0;
        let frr = same.iter().filter(|&&x| x > t).count() as f32 / same.len() as f32;
        let far = different.iter().filter(|&&x| x <= t).count() as f32 / different.len() as f32;
        if (far - frr).abs() < best_gap {
            best_gap = (far - frr).abs();
            best_t = t;
        }
    }
    best_t
}

/// Best-accuracy threshold from a sweep over observed-distance midpoints.
pub fn best_threshold(same: &[f32], different: &[f32]) -> (f32, f32) {
    let mut candidates: Vec<f32> = same.iter().chain(different.iter()).copied().collect();
    candidates.sort_by(f32::total_cmp);
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let mut best_t = 0.0;
    let mut best_acc = -1.0;
    for w in candidates.windows(2) {
        let t = (w[0] + w[1]) / 2.0;
        let acc = pair_accuracy(t, same, different);
        if acc > best_acc {
            best_acc = acc;
            best_t = t;
        }
    }
    (best_t, best_acc)
}

#[derive(Debug)]
pub struct Stats {
    pub mean: f32,
    pub p05: f32,
    pub p50: f32,
    pub p95: f32,
    pub min: f32,
    pub max: f32,
}

impl Stats {
    pub fn of(values: &[f32]) -> Stats {
        let mut v: Vec<f32> = values.to_vec();
        v.sort_by(f32::total_cmp);
        let pct = |p: f32| {
            let idx = ((v.len() as f32 - 1.0) * p).round() as usize;
            v[idx.min(v.len() - 1)]
        };
        Stats {
            mean: v.iter().sum::<f32>() / v.len() as f32,
            p05: pct(0.05),
            p50: pct(0.50),
            p95: pct(0.95),
            min: v[0],
            max: v[v.len() - 1],
        }
    }
}
