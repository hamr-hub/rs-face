//! rs-face-detect — face detection CLI that exercises both CPU and GPU
//! backends on the same frames.
//!
//! Why this binary exists
//! ----------------------
//! The core detection algorithm is in Rust; the user-visible Python
//! pipeline (``tools/annotate_all_faces.py``) only annotates the boxes.
//! This binary emits the same JSONL schema with **two** detection sets
//! per frame — one from the pure-Rust CPU cascade and one from the
//! selected GPU backend. By construction they must come back identical
//! (same cascade weights, same integral image, same variance
//! normalisation); the ``tools/compare_cpu_gpu.py`` auxiliary verifies
//! that on real videos.
//!
//! Usage
//! -----
//! ::
//!
//!     # Use the first available GPU backend; also run CPU cascade for parity.
//!     rs-face-detect <video.mp4> --out <dir>
//!
//!     # Force a specific backend (cpu, opencl, metal, cuda, rocm, ascend, mlu).
//!     rs-face-detect <video.mp4> --backend opencl --out <dir>
//!
//!     # CPU-only (skip GPU entirely).
//!     rs-face-detect <video.mp4> --backend cpu --out <dir>
//!
//! Output
//! ------
//! ``<out>/<stem>/detections.jsonl`` — one JSON record per sampled frame:
//!
//! .. code-block:: json
//!
//!     {
//!       "video": "<stem>",
//!       "frame_index": 0,
//!       "timestamp_ms": 0,
//!       "boxes_cpu":  [{"x": x, "y": y, "w": w, "h": h, "score": s}, ...],
//!       "boxes_gpu":  [{"x": x, "y": y, "w": w, "h": h, "score": s}, ...],
//!       "backend":    "opencl"
//!     }
//!
//! ``boxes_cpu`` is always present; ``boxes_gpu`` is only present when
//! the requested backend probed successfully.

use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rsface::detector::{non_max_suppression, Detection, Detector, DetectorConfig};
use rsface::gpu::backend::{self as gpu_backend, GpuBackend};
use rsface::haar::{params::demo_face_cascade, Cascade};
use rsface::image::GrayImage;
use rsface::source::{FfmpegPipeSource, FrameSource};

/// Parsed CLI arguments. Pure data — extracted from `parse_args` so the
/// parsing logic can be unit-tested without spawning the binary.
#[derive(Debug, PartialEq)]
struct ParsedArgs {
    input_path: PathBuf,
    out_dir: PathBuf,
    backend_arg: String,
    sample_fps: u32,
    max_frames: Option<usize>,
    min_size: usize,
    scale_factor: f32,
    stride: usize,
    skip_baseline: bool,
}

/// Argument-parser outcomes. `ShowHelp` is distinct from the error variants
/// because the binary should exit 0 (graceful) when the user asked for help,
/// versus exit 2 (error) when the input was malformed.
#[derive(Debug, PartialEq)]
enum ParseError {
    ShowHelp,
    MissingInput,
    MissingOut,
    UnknownArgument(String),
}

/// Parse CLI arguments into a `ParsedArgs`. Returns `Err(ParseError::ShowHelp)`
/// when `--help` / `-h` is present or no args were supplied; `Err` for the
/// remaining error variants otherwise.
fn parse_args(args: &[String]) -> Result<ParsedArgs, ParseError> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        return Err(ParseError::ShowHelp);
    }
    let mut input_path: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut backend_arg = "auto".to_string();
    let mut sample_fps: u32 = 5;
    let mut max_frames: Option<usize> = None;
    let mut min_size: usize = 24;
    let mut scale_factor: f32 = 1.2;
    let mut stride: usize = 4;
    let mut skip_baseline = false;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--out" => {
                out_dir = args.get(i + 1).map(PathBuf::from);
                i += 1;
            }
            "--backend" => {
                backend_arg = args.get(i + 1).cloned().unwrap_or("auto".into());
                i += 1;
            }
            "--sample-fps" => {
                sample_fps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(5);
                i += 1;
            }
            "--max-frames" => {
                max_frames = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            "--min-size" => {
                min_size = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(24);
                i += 1;
            }
            "--scale" => {
                scale_factor = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1.2);
                i += 1;
            }
            "--stride" => {
                stride = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(4);
                i += 1;
            }
            "--cpu-only" => {
                skip_baseline = false;
                backend_arg = "cpu".into();
            }
            "--gpu-only" => {
                skip_baseline = true;
            }
            other if !other.starts_with("--") && input_path.is_none() => {
                input_path = Some(PathBuf::from(other));
            }
            other => {
                return Err(ParseError::UnknownArgument(other.to_string()));
            }
        }
        i += 1;
    }

    let input_path = input_path.ok_or(ParseError::MissingInput)?;
    let out_dir = out_dir.ok_or(ParseError::MissingOut)?;
    Ok(ParsedArgs {
        input_path,
        out_dir,
        backend_arg,
        sample_fps,
        max_frames,
        min_size,
        scale_factor,
        stride,
        skip_baseline,
    })
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let parsed = match parse_args(&args) {
        Ok(p) => p,
        Err(ParseError::ShowHelp) => {
            print_help();
            return;
        }
        Err(ParseError::MissingInput) => {
            eprintln!("missing <INPUT>");
            print_help();
            std::process::exit(2);
        }
        Err(ParseError::MissingOut) => {
            eprintln!("missing --out <DIR>");
            print_help();
            std::process::exit(2);
        }
        Err(ParseError::UnknownArgument(name)) => {
            eprintln!("unknown argument: {}", name);
            print_help();
            std::process::exit(2);
        }
    };
    let ParsedArgs {
        input_path,
        out_dir,
        backend_arg,
        sample_fps,
        max_frames,
        min_size,
        scale_factor,
        stride,
        skip_baseline,
    } = parsed;
    if !input_path.exists() {
        eprintln!("input file not found: {}", input_path.display());
        std::process::exit(2);
    }

    // Probe the GPU backend early so we can fail fast and print what we
    // actually got.
    let backend: Option<Box<dyn GpuBackend>> = if backend_arg == "cpu" {
        None
    } else if backend_arg == "auto" {
        gpu_backend::auto()
    } else {
        gpu_backend::get(&backend_arg)
    };
    let backend_label = match &backend {
        Some(b) => b.info().one_line(),
        None => "<no GPU backend>".to_string(),
    };
    println!("== rs-face-detect ==");
    println!("  input  : {}", input_path.display());
    println!("  out    : {}", out_dir.display());
    println!("  backend: {} ({})", backend_arg, backend_label);
    println!("  fps    : {}", sample_fps);
    println!("  stride : {}  min_size: {}", stride, min_size);

    let cascade: Cascade = demo_face_cascade();
    println!(
        "  cascade: built-in demo ({} stages, {} features, {}x{} window)",
        cascade.num_stages(),
        cascade.num_features(),
        cascade.window_w,
        cascade.window_h
    );

    let det_cfg = DetectorConfig {
        min_size,
        max_size: 1024,
        scale_factor,
        window_stride: stride,
        nms_iou_threshold: 0.3,
        min_score: 0.0,
        variance_threshold: 200,
        use_gpu: false,
        equalize_hist: false,
    };
    let nms_iou = det_cfg.nms_iou_threshold;
    let detector = Detector::new(cascade.clone(), det_cfg);

    // Open the video via ffmpeg. The pipe source samples at the requested
    // fps and downscaled to 480 wide, matching the production pipeline.
    let mut src = match FfmpegPipeSource::new(input_path.to_str().unwrap(), sample_fps) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to open video: {}", e);
            std::process::exit(3);
        }
    };

    let stem = input_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "video".into());
    let video_out = out_dir.join(&stem);
    fs::create_dir_all(&video_out).expect("create out dir");

    let det_path = video_out.join("detections.jsonl");
    let mut det_f = BufWriter::new(fs::File::create(&det_path).expect("open jsonl"));

    let mut n_frames = 0usize;
    let mut n_boxes_cpu = 0usize;
    let mut n_boxes_gpu = 0usize;
    let mut total_wall_ms: f64 = 0.0;
    let t_start = Instant::now();

    while let Some(frame) = src.next_frame().expect("frame decode") {
        if let Some(m) = max_frames {
            if n_frames >= m {
                break;
            }
        }
        let gray = frame.gray.clone();
        let img_ref: &GrayImage = &gray;

        // --- CPU pass
        let t_cpu = Instant::now();
        let mut cpu_dets = detector.detect(img_ref);
        cpu_dets = non_max_suppression(cpu_dets, nms_iou);
        let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;
        n_boxes_cpu += cpu_dets.len();

        // --- GPU pass (if backend available)
        let (gpu_dets, gpu_ms) = if let Some(b) = backend.as_ref() {
            let t = Instant::now();
            let raw = b.detect_windows(&cascade, img_ref, 4096);
            let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
            // GPU returns (x, y, score). Convert to (x, y, window_w, window_h, score).
            let boxes: Vec<(i32, i32, i32, i32, f32)> = raw
                .iter()
                .map(|d| {
                    // Use the (w, h) reported by the backend; the OpenCL
                    // passthrough fills 0 (the caller patches with the
                    // cascade window), so use the cascade window as the
                    // fallback.
                    let w = if d.w > 0 {
                        d.w as i32
                    } else {
                        cascade.window_w as i32
                    };
                    let h = if d.h > 0 {
                        d.h as i32
                    } else {
                        cascade.window_h as i32
                    };
                    (d.x as i32, d.y as i32, w, h, d.score)
                })
                .collect();
            // NMS using the same IoU threshold as the CPU pass.
            let mut nms_dets: Vec<Detection> = boxes
                .iter()
                .map(|&(x, y, w, h, s)| Detection {
                    x: x as usize,
                    y: y as usize,
                    w: w as usize,
                    h: h as usize,
                    score: s,
                })
                .collect();
            nms_dets = non_max_suppression(nms_dets, nms_iou);
            (nms_dets, elapsed_ms)
        } else {
            (Vec::new(), 0.0)
        };
        n_boxes_gpu += gpu_dets.len();

        // --- Write JSONL row.
        writeln!(
            det_f,
            "{}",
            serde_json_like(&FrameRec {
                video: &stem,
                frame_index: n_frames,
                timestamp_ms: frame.timestamp_ms,
                boxes_cpu: &cpu_dets,
                boxes_gpu: &gpu_dets,
                backend: backend_label.as_str(),
                cpu_ms,
                gpu_ms,
            })
        )
        .expect("write jsonl");
        total_wall_ms += cpu_ms + gpu_ms;

        n_frames += 1;
        if n_frames % 10 == 0 {
            eprint!("  ... {} frames\r", n_frames);
        }
    }
    det_f.flush().ok();

    let wall = t_start.elapsed().as_secs_f64();
    eprintln!();
    println!("== summary ==");
    println!("  frames   : {}", n_frames);
    println!("  cpu boxes: {}", n_boxes_cpu);
    println!("  gpu boxes: {}", n_boxes_gpu);
    println!("  wall     : {:.2}s (decode+detect)", wall);
    println!(
        "  per-f CPU: {:.1}ms",
        if n_frames > 0 {
            total_wall_ms / n_frames as f64
        } else {
            0.0
        }
    );
    println!("  jsonl    : {}", det_path.display());
}

fn print_help() {
    println!(
        "rs-face-detect — CPU + GPU face detector with bit-identical output\n\n\
         USAGE:\n\
           rs-face-detect <INPUT> --out <DIR> [options]\n\n\
         INPUT:\n\
           /path/to/video.mp4 (decoded via ffmpeg on PATH)\n\n\
         OPTIONS:\n\
           --out <DIR>           output directory (required)\n\
           --backend <ID>        cpu | opencl | metal | cuda | rocm | ascend | mlu | auto\n\
                                 (default: auto — picks first available)\n\
           --sample-fps <N>      sample at N fps (default 5)\n\
           --max-frames <N>      stop after N frames\n\
           --min-size <PX>       minimum detection size (default 24)\n\
           --scale <F>           pyramid scale factor (default 1.2)\n\
           --stride <PX>         window stride (default 4)\n\
           --cpu-only            skip the GPU pass\n\
           --gpu-only            skip the CPU pass (verification only)\n\
           --help                print this help\n"
    );
}

// ---------- minimal JSON writer (avoids pulling in serde) ----------

struct FrameRec<'a> {
    video: &'a str,
    frame_index: usize,
    timestamp_ms: u64,
    boxes_cpu: &'a [Detection],
    boxes_gpu: &'a [Detection],
    backend: &'a str,
    cpu_ms: f64,
    gpu_ms: f64,
}

fn serde_json_like(rec: &FrameRec<'_>) -> String {
    let mut s = String::with_capacity(1024);
    s.push('{');
    kv(&mut s, "video", &json_str(rec.video), false);
    s.push(',');
    kv_num(&mut s, "frame_index", rec.frame_index as f64, false);
    s.push(',');
    kv_num(&mut s, "timestamp_ms", rec.timestamp_ms as f64, false);
    s.push(',');
    boxes_json(&mut s, "boxes_cpu", rec.boxes_cpu);
    s.push(',');
    boxes_json(&mut s, "boxes_gpu", rec.boxes_gpu);
    s.push(',');
    kv(&mut s, "backend", &json_str(rec.backend), false);
    s.push(',');
    kv_num(&mut s, "cpu_ms", rec.cpu_ms, false);
    s.push(',');
    kv_num(&mut s, "gpu_ms", rec.gpu_ms, false);
    s.push('}');
    s
}

fn kv(s: &mut String, key: &str, val: &str, quote: bool) {
    s.push('"');
    s.push_str(key);
    s.push('"');
    s.push(':');
    if quote {
        s.push('"');
        s.push_str(val);
        s.push('"');
    } else {
        s.push_str(val);
    }
}

fn kv_num(s: &mut String, key: &str, val: f64, _: bool) {
    s.push('"');
    s.push_str(key);
    s.push('"');
    s.push(':');
    if val.fract() == 0.0 && val.is_finite() && val.abs() < 1e15 {
        s.push_str(&format!("{}", val as i64));
    } else {
        s.push_str(&format!("{:.4}", val));
    }
}

fn boxes_json(s: &mut String, key: &str, boxes: &[Detection]) {
    s.push('"');
    s.push_str(key);
    s.push('"');
    s.push(':');
    s.push('[');
    for (i, b) in boxes.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('{');
        s.push_str(&format!(
            "\"x\":{},\"y\":{},\"w\":{},\"h\":{},\"score\":{:.4}",
            b.x, b.y, b.w, b.h, b.score
        ));
        s.push('}');
    }
    s.push(']');
}

fn json_str(s_in: &str) -> String {
    let mut out = String::with_capacity(s_in.len() + 2);
    out.push('"');
    for c in s_in.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `print_help` is called from multiple error paths; verify it
    /// doesn't panic (e.g. no dangling `{` in the format string).
    #[test]
    fn print_help_does_not_panic() {
        print_help();
    }

    /// Empty argv + `--help` + `-h` should all map to `ShowHelp`, not to
    /// `MissingInput` — the binary must exit 0 (graceful) when the user
    /// asked for help.
    #[test]
    fn empty_args_or_help_returns_showhelp() {
        assert_eq!(parse_args(&[]).unwrap_err(), ParseError::ShowHelp);
        assert_eq!(
            parse_args(&["--help".into()]).unwrap_err(),
            ParseError::ShowHelp
        );
        assert_eq!(
            parse_args(&["-h".into()]).unwrap_err(),
            ParseError::ShowHelp
        );
    }

    /// `<INPUT>` without `--out <DIR>` must fail with `MissingOut`, not
    /// `MissingInput`. Guards the order of error reporting and the CLI UX
    /// (telling the user what they did supply vs. what's missing).
    #[test]
    fn missing_out_dir_returns_missing_out() {
        let args = vec!["video.mp4".to_string()];
        match parse_args(&args) {
            Err(ParseError::MissingOut) => {} // expected
            other => panic!("expected MissingOut, got {:?}", other),
        }
    }

    /// Bare argv (no `<INPUT>` at all) must fail with `MissingInput`.
    #[test]
    fn missing_input_returns_missing_input() {
        let args = vec!["--out".to_string(), "out".to_string()];
        match parse_args(&args) {
            Err(ParseError::MissingInput) => {} // expected
            other => panic!("expected MissingInput, got {:?}", other),
        }
    }

    /// Unknown flags like `--frobnicate` must come back as
    /// `UnknownArgument` so the binary can route them to `exit(2)`.
    #[test]
    fn unknown_flag_returns_unknown_argument() {
        let args = vec![
            "video.mp4".to_string(),
            "--out".to_string(),
            "out".to_string(),
            "--frobnicate".to_string(),
        ];
        match parse_args(&args) {
            Err(ParseError::UnknownArgument(name)) => {
                assert_eq!(name, "--frobnicate");
            }
            other => panic!("expected UnknownArgument, got {:?}", other),
        }
    }

    /// Happy-path parse: a minimal valid invocation must populate every
    /// field on `ParsedArgs` with the expected defaults / supplied values.
    /// Defaults come from the binary's "good citizen" production defaults.
    #[test]
    fn happy_path_parses_all_fields() {
        // Use a guaranteed-existing argv slot: the parser doesn't validate
        // file existence (that's done in `main` after parsing), so any
        // path is fine here.
        let args = vec![
            "video.mp4".to_string(),
            "--out".to_string(),
            "outdir".to_string(),
            "--backend".to_string(),
            "cuda".to_string(),
            "--sample-fps".to_string(),
            "10".to_string(),
            "--max-frames".to_string(),
            "42".to_string(),
            "--min-size".to_string(),
            "32".to_string(),
            "--scale".to_string(),
            "1.1".to_string(),
            "--stride".to_string(),
            "6".to_string(),
        ];
        let p = parse_args(&args).expect("valid args");
        assert_eq!(p.input_path, PathBuf::from("video.mp4"));
        assert_eq!(p.out_dir, PathBuf::from("outdir"));
        assert_eq!(p.backend_arg, "cuda");
        assert_eq!(p.sample_fps, 10);
        assert_eq!(p.max_frames, Some(42));
        assert_eq!(p.min_size, 32);
        assert!((p.scale_factor - 1.1).abs() < 1e-6);
        assert_eq!(p.stride, 6);
        assert!(!p.skip_baseline); // no --gpu-only
    }

    /// `--gpu-only` is the only flag that mutates `skip_baseline`. Default
    /// is false (CPU pass runs alongside GPU pass for parity check).
    #[test]
    fn gpu_only_flag_sets_skip_baseline() {
        let args = vec![
            "video.mp4".to_string(),
            "--out".to_string(),
            "outdir".to_string(),
            "--gpu-only".to_string(),
        ];
        let p = parse_args(&args).expect("valid args");
        assert!(p.skip_baseline, "--gpu-only should set skip_baseline=true");
        // --gpu-only does NOT touch backend_arg (unlike --cpu-only which
        // forces "cpu"); the user can still pick a specific GPU backend.
        assert_eq!(p.backend_arg, "auto");
    }

    /// `--cpu-only` is shorthand for `--backend cpu`. The two must agree.
    #[test]
    fn cpu_only_flag_forces_backend_cpu() {
        let args = vec![
            "video.mp4".to_string(),
            "--out".to_string(),
            "outdir".to_string(),
            "--cpu-only".to_string(),
        ];
        let p = parse_args(&args).expect("valid args");
        assert_eq!(p.backend_arg, "cpu");
        assert!(
            !p.skip_baseline,
            "--cpu-only runs CPU pass alongside (no GPU)"
        );
    }

    /// Numeric parsing must fall back to defaults on garbage, not crash.
    #[test]
    fn malformed_numeric_args_fall_back_to_defaults() {
        let args = vec![
            "video.mp4".to_string(),
            "--out".to_string(),
            "outdir".to_string(),
            "--sample-fps".to_string(),
            "not-a-number".to_string(),
            "--min-size".to_string(),
            "also-not-a-number".to_string(),
            "--scale".to_string(),
            "garbage".to_string(),
        ];
        let p = parse_args(&args).expect("valid arg layout");
        // Each malformed value must default cleanly.
        assert_eq!(p.sample_fps, 5);
        assert_eq!(p.min_size, 24);
        assert!((p.scale_factor - 1.2).abs() < 1e-6);
    }
}
