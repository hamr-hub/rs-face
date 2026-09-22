//! Command-line silent face-anti-spoofing check and offline collector.
//!
//! Standalone companion to the platform liveness integration. For one image
//! it reports whether the largest face looks live or like a print / screen
//! replay. Pointing it at a directory turns it into an offline collector: it
//! scans every supported image once, prints a verdict per file and finishes
//! with an aggregate summary (real/spoof counts and mean quality signals),
//! which is the offline way to gather calibration data without a server or a
//! database. Like every bin touching inference graphs, this is **not** part
//! of the zero-dependency story: it is gated behind an ONNX backend feature
//! and real weights.
//!
//! Pipeline per file: decode → largest face with SCRFD → two-model MiniFASNet
//! check → verdict. Single-image mode exits non-zero on spoof/error so it
//! composes in a shell; directory mode is report-only and exits zero when all
//! files were processed.
//!
//! ```sh
//! tools/fetch_models.sh
//! cargo run --release --features tract-backend --bin liveness_check -- \
//!     --models models photo.jpg
//! cargo run --release --features tract-backend --bin liveness_check -- \
//!     --models models ./capture_dir
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rsface::liveness::{LivenessConfig, LivenessOutcome};
use rsface::liveness_detector::LivenessDetector;
use rsface::models::{self, SCRFD_10G_KPS};
use rsface::onnx::SessionConfig;
use rsface::quality::QualityConfig;
use rsface::scrfd::ScrfdConfig;
use rsface::scrfd_detector::ScrfdDetector;

struct Cli {
    models_dir: PathBuf,
    target: PathBuf,
    min_real_score: f32,
    quality_gate: bool,
}

fn print_usage() {
    eprintln!(
        "usage: liveness_check [--models <DIR>] [--min-real-score F] [--quality-gate] <IMAGE|DIR>\n\
         single image: exit 0 live / 1 spoof or error\n\
         directory:    scan all images, report aggregate, exit 0 when all processed"
    );
}

fn parse_args() -> Result<Cli, String> {
    let mut models_dir = PathBuf::from("models");
    let mut target: Option<PathBuf> = None;
    let mut min_real_score = 0.0f32;
    let mut quality_gate = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--models" => {
                models_dir = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--models needs a directory".to_string())?,
                );
            }
            "--min-real-score" => {
                min_real_score = args
                    .next()
                    .ok_or_else(|| "--min-real-score needs a value".to_string())?
                    .parse()
                    .map_err(|_| "invalid --min-real-score".to_string())?;
            }
            "--quality-gate" => quality_gate = true,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other if !other.starts_with("--") && target.is_none() => {
                target = Some(PathBuf::from(other));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Cli {
        models_dir,
        target: target.ok_or_else(|| "missing IMAGE|DIR".to_string())?,
        min_real_score,
        quality_gate,
    })
}

/// Loaded once and reused for every file in a batch.
struct Engines {
    faces: ScrfdDetector,
    liveness: LivenessDetector,
}

fn decode_image(path: &Path) -> Result<rsface::image::RgbImage, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => rsface::image::png::decode_to_rgb(&mut std::io::Cursor::new(bytes))
            .map_err(|e| format!("decode png: {e}")),
        Some("jpg" | "jpeg") => {
            rsface::image::jpeg::decode_jpeg_rgb(&bytes).map_err(|e| format!("decode jpeg: {e:?}"))
        }
        Some("ppm") => rsface::image::codec::read_ppm(&mut std::io::Cursor::new(bytes))
            .map_err(|e| format!("decode ppm: {e}")),
        other => Err(format!("unsupported image type: {other:?}")),
    }
}

fn resolve(models_dir: &Path, file: &str) -> Result<PathBuf, String> {
    let p = models_dir.join(file);
    if p.exists() {
        Ok(p)
    } else {
        Err(format!(
            "missing {} (run tools/fetch_models.sh)",
            p.display()
        ))
    }
}

fn load_engines(cli: &Cli) -> Result<Engines, String> {
    let detector_path = resolve(&cli.models_dir, "det_10g.onnx")?;
    let face_detector = ScrfdDetector::open(
        &detector_path,
        Some(&SCRFD_10G_KPS),
        &SessionConfig::default(),
        ScrfdConfig::default(),
    )
    .map_err(|e| format!("load SCRFD: {e:?}"))?;

    let path_v2 = resolve(&cli.models_dir, models::LIVENESS_MINIFASNET_V2.file_name)?;
    let path_v1se = resolve(&cli.models_dir, models::LIVENESS_MINIFASNET_V1SE.file_name)?;
    let quality = if cli.quality_gate {
        QualityConfig::strict()
    } else {
        QualityConfig::default()
    };
    let liveness = LivenessDetector::open(
        &path_v2,
        Some(&models::LIVENESS_MINIFASNET_V2),
        &path_v1se,
        Some(&models::LIVENESS_MINIFASNET_V1SE),
        &SessionConfig::default(),
        LivenessConfig {
            min_real_score: cli.min_real_score,
        },
        quality,
    )
    .map_err(|e| format!("load liveness: {e:?}"))?;

    Ok(Engines {
        faces: face_detector,
        liveness,
    })
}

/// Running aggregate of verdicts and quality signals across a batch.
#[derive(Default)]
struct Aggregate {
    processed: u64,
    real: u64,
    spoof: u64,
    no_face: u64,
    sharpness_sum: f64,
    brightness_sum: f64,
    clipped_sum: f64,
    high_freq_sum: f64,
    signal_n: u64,
}

impl Aggregate {
    fn add(&mut self, out: &LivenessOutcome) {
        self.processed += 1;
        if out.is_real {
            self.real += 1;
        } else {
            self.spoof += 1;
        }
        if let Some(q) = &out.quality {
            self.sharpness_sum += q.sharpness as f64;
            self.brightness_sum += q.mean_brightness as f64;
            self.clipped_sum += q.clipped_ratio as f64;
            self.high_freq_sum += q.high_freq_ratio as f64;
            self.signal_n += 1;
        }
    }

    fn mean(&self, sum: f64) -> Option<f32> {
        (self.signal_n > 0).then(|| (sum / self.signal_n as f64) as f32)
    }

    fn print_summary(&self) {
        println!(
            "aggregate: processed={} real={} spoof={} no_face={}",
            self.processed, self.real, self.spoof, self.no_face
        );
        if self.signal_n > 0 {
            println!(
                "mean signals (n={}): sharpness={:.2} brightness={:.2} clipped={:.3} high_freq={:.3}",
                self.signal_n,
                self.mean(self.sharpness_sum).unwrap_or(0.0),
                self.mean(self.brightness_sum).unwrap_or(0.0),
                self.mean(self.clipped_sum).unwrap_or(0.0),
                self.mean(self.high_freq_sum).unwrap_or(0.0),
            );
        }
    }
}

/// `Ok(None)` means the image was processed but contained no face — a normal
/// outcome in a capture directory, not an error.
fn check_one(
    engines: &Engines,
    img: &rsface::image::RgbImage,
) -> Result<Option<LivenessOutcome>, String> {
    let faces = engines
        .faces
        .detect_rgb(img)
        .map_err(|e| format!("face detection: {e:?}"))?;
    let Some(largest) = faces.into_iter().max_by(|a, b| {
        a.area()
            .partial_cmp(&b.area())
            .unwrap_or(std::cmp::Ordering::Equal)
    }) else {
        return Ok(None);
    };
    engines
        .liveness
        .check(img, &largest.to_detection())
        .map(Some)
        .map_err(|e| format!("liveness check: {e:?}"))
}

fn print_line(path: &Path, out: &LivenessOutcome) {
    let quality_note = out
        .quality
        .as_ref()
        .map(|q| {
            format!(
                " sharpness={:.1} high_freq={:.3}",
                q.sharpness, q.high_freq_ratio
            )
        })
        .unwrap_or_default();
    println!(
        "{}: is_real={} label=\"{}\" real_score={:.4} probs={:.3}/{:.3}/{:.3}{quality_note}",
        path.display(),
        out.is_real,
        out.label(),
        out.real_score,
        out.probs[0],
        out.probs[1],
        out.probs[2]
    );
}

/// Files to scan when the target is a directory: supported images at the top
/// level, sorted for deterministic output.
fn collect_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let supported = ["png", "jpg", "jpeg", "ppm"];
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| supported.contains(&e.to_ascii_lowercase().as_str()))
        })
        .collect();
    files.sort();
    files
}

fn main() -> ExitCode {
    let cli = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    let engines = match load_engines(&cli) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };

    if cli.target.is_dir() {
        let files = collect_files(&cli.target);
        if files.is_empty() {
            eprintln!("error: no supported images in {}", cli.target.display());
            return ExitCode::from(1);
        }
        let mut agg = Aggregate::default();
        let mut failures = 0u64;
        for file in &files {
            match decode_image(file) {
                Ok(img) => match check_one(&engines, &img) {
                    Ok(Some(out)) => {
                        print_line(file, &out);
                        agg.add(&out);
                    }
                    Ok(None) => {
                        eprintln!("{}: no face detected", file.display());
                        failures += 1;
                    }
                    Err(e) => {
                        eprintln!("{}: {e}", file.display());
                        failures += 1;
                    }
                },
                Err(e) => {
                    eprintln!("{e}");
                    failures += 1;
                }
            }
        }
        agg.print_summary();
        // Report-only: success when every file was processed.
        if failures == 0 {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        }
    } else {
        let result = decode_image(&cli.target)
            .map_err(|e| format!("error: {e}"))
            .and_then(|img| check_one(&engines, &img).map_err(|e| format!("error: {e}")));
        match result {
            Ok(Some(out)) => {
                print_line(&cli.target, &out);
                if out.is_real {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(1)
                }
            }
            Ok(None) => {
                eprintln!("error: no face detected in {}", cli.target.display());
                ExitCode::from(2)
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::from(1)
            }
        }
    }
}
