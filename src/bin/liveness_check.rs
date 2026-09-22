//! Command-line silent face-anti-spoofing check.
//!
//! Standalone companion to the platform liveness integration. It answers, for
//! one image file, whether the detected face looks live or like a print /
//! screen replay, without spinning up the server. Like every bin touching
//! inference graphs, this is **not** part of the zero-dependency story: it is
//! gated behind an ONNX backend feature and real weights.
//!
//! Pipeline: decode the image → find the largest face with SCRFD → expand that
//! box for MiniFASNet → run the two-model check → print the verdict and exit
//! non-zero when the face is not accepted as live, so the tool composes in a
//! shell (`if liveness_check photo.jpg; then ...`).
//!
//! ```sh
//! tools/fetch_models.sh
//! cargo run --release --features tract-backend --bin liveness_check -- \
//!     --models models photo.jpg
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use rsface::liveness::{LivenessConfig, LivenessOutcome};
use rsface::liveness_detector::LivenessDetector;
use rsface::models::{self, ModelSpec, SCRFD_10G_KPS};
use rsface::onnx::SessionConfig;
use rsface::quality::QualityConfig;
use rsface::scrfd::ScrfdConfig;
use rsface::scrfd_detector::ScrfdDetector;

struct Cli {
    models_dir: PathBuf,
    image: PathBuf,
    min_real_score: f32,
    quality_gate: bool,
}

fn print_usage() {
    eprintln!(
        "usage: liveness_check [--models <DIR>] [--min-real-score F] [--quality-gate] <IMAGE>\n\
         exit 0 when the largest face is accepted live, 1 on spoof/error"
    );
}

fn parse_args() -> Result<Cli, String> {
    let mut models_dir = PathBuf::from("models");
    let mut image: Option<PathBuf> = None;
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
            other if !other.starts_with("--") && image.is_none() => {
                image = Some(PathBuf::from(other));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Cli {
        models_dir,
        image: image.ok_or_else(|| "missing IMAGE".to_string())?,
        min_real_score,
        quality_gate,
    })
}

fn decode_image(path: &std::path::Path) -> Result<rsface::image::RgbImage, String> {
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

fn resolve(models_dir: &std::path::Path, file: &str) -> Result<PathBuf, String> {
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

fn run(cli: &Cli) -> Result<LivenessOutcome, String> {
    let img = decode_image(&cli.image)?;

    let detector_path = resolve(&cli.models_dir, "det_10g.onnx")?;
    let face_detector = ScrfdDetector::open(
        &detector_path,
        Some(&SCRFD_10G_KPS),
        &SessionConfig::default(),
        ScrfdConfig::default(),
    )
    .map_err(|e| format!("load SCRFD: {e:?}"))?;

    let faces = face_detector
        .detect_rgb(&img)
        .map_err(|e| format!("face detection: {e:?}"))?;
    let largest = faces
        .into_iter()
        .max_by(|a, b| {
            a.area()
                .partial_cmp(&b.area())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| "no face found".to_string())?;
    let detection = largest.to_detection();

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

    liveness
        .check(&img, &detection)
        .map_err(|e| format!("liveness check: {e:?}"))
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
    match run(&cli) {
        Ok(out) => {
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
                cli.image.display(),
                out.is_real,
                out.label(),
                out.real_score,
                out.probs[0],
                out.probs[1],
                out.probs[2]
            );
            if out.is_real {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
