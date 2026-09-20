//! rs-face CLI — face detection on a video URL or image sequence.

use std::path::PathBuf;
use std::time::Instant;

use rsface::detector::{Detector, DetectorConfig};
use rsface::face_detector::FaceDetector;
use rsface::haar::bundled::bundled_frontalface_cascade;
use rsface::haar::Cascade;
use rsface::image::GrayImage;
use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};
use rsface::pipeline::{Pipeline, PipelineConfig};
use rsface::source;

fn print_help() {
    println!(
        "rs-face — zero-dep multi-algorithm face detector\n\n\
         USAGE:\n  \
         rs-face demo                         zero-arg install check on a built-in portrait\n  \
         rs-face <INPUT> --out <DIR> --algo <haar|cnn|luminance> [options]\n\n\
         INPUT forms:\n  \
           demo                built-in 256x256 portrait, no external files (default --out ./rsface-demo)\n  \
           test://N            synthetic test pattern (N frames)\n  \
           /path/to/dir        image sequence (PNG/PPM/JPG files)\n  \
           /path/file.png|jpg  single image\n  \
           http(s)://host/p    single image or PNG sequence base URL\n  \
           *.mp4|*.mov|*.avi|*.mkv|*.webm | rtsp://...\n                           (requires `ffmpeg` on PATH)\n\n\
         ALGORITHMS:\n  \
           haar       Viola-Jones Haar cascade (default; core::Detector)\n  \
           cnn        tiny 24x24 Conv+ReLU+Pool+FC net, trainable with cnn_train\n  \
           luminance  band-pattern + mirror symmetry (no weights, classical CV)\n\n\
         OPTIONS:\n  \
           --out <DIR>           output directory (required)\n  \
           --algo <NAME>         detection algorithm (default: haar)\n  \
           --cascade <PATH>      load cascade from .rfcf file (haar only,\n  \
                                           default: bundled OpenCV frontalface cascade)\n  \
           --min-neighbors N     OpenCV minNeighbors: merge threshold for raw hits (default: 3;\n  \
                                           0 shows every raw window, matching cv::CascadeClassifier)\n  \
           --threads N           worker thread count (default: # CPUs)\n  \
           --min-size PX         minimum detection size in pixels (default: 24)\n  \
           --max-size PX         maximum detection size in pixels (default: 1024)\n  \
           --scale F             pyramid scale factor (default: 1.2)\n  \
           --stride PX           window stride in pixels (default: 4)\n  \
           --nms F               NMS IoU threshold (default: 0.3)\n  \
           --min-score F         drop detections with cascade score below this\n  \
           --only-with-face      skip writing frames with zero detections\n  \
           --queue-depth N       per-worker queue depth (default: 4)\n  \
           --cnn                 use the CNN face detector (default: built-in template weights)\n  \
           --cnn-weights PATH    load CNN weights from a .cnn.bin file (requires --cnn)\n  \
           --no-gpu              disable the GPU OpenCL backend\n  \
           --no-equalize         skip the cv::equalizeHist preprocessing\n  \
           --list-algos          list every algorithm with its maturity and description\n  \
           --list-features       list every Cargo feature this binary was compiled with\n  \
           --version             print the crate version and exit\n  \
           --help                print this help\n\n\
         RECIPES:\n  \
           # Zero-arg install check — real cascade on a built-in portrait:\n  \
           rs-face demo\n\n  \
           # Smoke test on a synthetic pattern:\n  \
           rs-face test://60 --out ./out\n\n  \
           # Real footage with the bundled cascade:\n  \
           rs-face video.mp4 --out ./out --threads 4\n\n  \
           # Real footage with a converted OpenCV Haar cascade:\n  \
           rs-face video.mp4 --out ./out --cascade haarcascade.rfcf\n\n  \
           # Heavy drama footage (variable face sizes):\n  \
           rs-face clip.mp4 --out ./out --scale 1.4 --stride 3 --only-with-face\n\n  \
           # Try the weight-free classical heuristic (band + symmetry):\n  \
           rs-face video.mp4 --out ./out --algo luminance\n\n  \
           # Industrial accuracy: SCRFD + ArcFace behind an ONNX feature —\n  \
           # see the detect_scrfd_arcface example and `--list-features`.\n\n  \
           # See also: `cargo run --example` for SDK recipes; docs/INDEX.md for the full doc map.\n"
    );
}

fn print_algos() {
    println!(
        "rs-face algorithms (compiled-in):\n\n  \
           name       maturity      description\n  \
           ---------  ------------  ----------------------------------------"
    );
    // The list is the source of truth. Adding a new detector means adding a row here.
    let rows: [(&str, &str, &str); 3] = [
        (
            "haar",
            "Production",
            "Viola-Jones AdaBoost cascade over 5 Haar-like feature families. Zero deps, real accuracy on frontal faces; load OpenCV XML via tools/convert_opencv_xml.py.",
        ),
        (
            "cnn",
            "Experimental",
            "Tiny 24x24 Conv+ReLU+Pool+FC net with a zero-dep trainer (cnn_train). Starter weights for smoke-testing; train your own and load them via --cnn-weights.",
        ),
        (
            "luminance",
            "Experimental",
            "Band-pattern + mirror-symmetry detector. No weights at all, fully classical CV; strongest on frontal portraits.",
        ),
    ];
    for (name, mat, desc) in rows {
        println!("  {:<9}  {:<12}  {}", name, mat, desc);
    }
    println!(
        "\nAlgorithm tags consumed by --algo and the RSFACE_ALGO env var.\n\
         Production:   measured accuracy in this crate (see docs/algorithms.md).\n\
         Experimental: runs end-to-end; accuracy not independently measured here.\n\
         Compile-time gate: the ort-backend / tract-backend features add a\n\
         SCRFD detector + ArcFace recogniser used through the library API and\n\
         examples (see --list-features)."
    );
}

fn print_features() {
    println!(
        "rs-face Cargo features compiled into this binary:\n\n  \
           (default)      Zero runtime deps. Pure-Rust CPU classical CV — Haar cascade,\n  \
                          luminance heuristic, tiny trainable CNN, and all three\n  \
                          zero-dep recognisers (LBPH, eigenfaces, Fisherfaces).\n\n  \
           metal-backend  Metal GPU on macOS / Apple Silicon (OpenCL path is deprecated on\n  \
                          current macOS). Off by default.\n\n  \
           cuda-backend   CUDA on Linux / Windows via cudarc 0.12. Needs CUDA toolkit +\n  \
                          NVIDIA driver at runtime. Off by default.\n\n  \
           ort-backend    ONNX Runtime (C++ runtime; GPU-capable: CoreML / CUDA / TensorRT /\n  \
                          DirectML / ROCm). Adds SCRFD detector + ArcFace recogniser.\n  \
                          Requires libonnxruntime.so on the host.\n\n  \
           tract-backend  Pure-Rust ONNX inference. No C++ toolchain, CPU-only, slower.\n  \
                          Same SCRFD + ArcFace model files as ort-backend.\n\n  \
           onnx           Alias for `ort-backend` (kept for ergonomics).\n\n  \
           ort-coreml / ort-cuda / ort-tensorrt / ort-directml\n  \
                          ONNX Runtime execution-provider toggles. Pair with ort-backend.\n\n\
         Examples:\n  \
           # smallest possible build (default features):\n  \
           cargo build --release\n\n  \
           # add Metal GPU on macOS:\n  \
           cargo build --release --features metal-backend\n\n  \
           # add industrial accuracy (ONNX Runtime, GPU-capable):\n  \
           cargo build --release --features ort-backend\n\n  \
           # add industrial accuracy without C++ (pure Rust, CPU only):\n  \
           cargo build --release --features tract-backend"
    );
}

/// Best-match suggestion for a typo'd algorithm name. Returns the closest
/// known name within edit-distance ≤ 3, or `None` if everything is far.
fn did_you_mean<'a>(needle: &str, haystack: &'a [&'a str]) -> Option<&'a str> {
    let mut best: Option<(&'a str, usize)> = None;
    for &cand in haystack {
        let d = edit_distance(needle, cand);
        if d <= 3 && best.map_or(true, |(_, bd)| d < bd) {
            best = Some((cand, d));
        }
    }
    best.map(|(c, _)| c)
}

/// Iterative Damerau-style edit distance (insert / delete / substitute /
/// adjacent transposition). Small enough for the 8-element algorithm list;
/// a real CLI parser would swap in a published crate.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev2 = vec![0usize; m + 1];
    let mut prev1 = vec![0usize; m + 1];
    let mut curr = vec![0usize; m + 1];
    for j in 0..=m {
        prev1[j] = j;
    }
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = std::cmp::min(
                std::cmp::min(curr[j - 1] + 1, prev1[j] + 1),
                prev1[j - 1] + cost,
            );
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                curr[j] = std::cmp::min(curr[j], prev2[j - 2] + 1);
            }
        }
        std::mem::swap(&mut prev2, &mut prev1);
        std::mem::swap(&mut prev1, &mut curr);
    }
    prev1[m]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut input: Option<String> = None;
    let mut out: Option<PathBuf> = None;
    let mut cascade_path: Option<PathBuf> = None;
    let mut threads: Option<usize> = None;
    let mut queue_depth: Option<usize> = None;
    let mut min_size: Option<usize> = None;
    let mut max_size: Option<usize> = None;
    let mut scale: Option<f32> = None;
    let mut stride: Option<usize> = None;
    let mut nms: Option<f32> = None;
    let mut min_neighbors: Option<i32> = None;
    let mut min_score: Option<f32> = None;
    let mut only_with_face = false;
    let mut no_gpu = false;
    let mut no_equalize = false;
    let mut use_cnn = false;
    let mut cnn_weights_path: Option<PathBuf> = None;
    let mut algo: Option<String> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--help" | "-h" => {
                print_help();
                return;
            }
            "--version" | "-V" => {
                println!("rs-face {}", env!("CARGO_PKG_VERSION"));
                return;
            }
            "--list-algos" => {
                print_algos();
                return;
            }
            "--list-features" => {
                print_features();
                return;
            }
            "--out" => {
                out = args.next().map(PathBuf::from);
            }
            "--cascade" => {
                cascade_path = args.next().map(PathBuf::from);
            }
            "--threads" => {
                threads = args.next().and_then(|s| s.parse().ok());
            }
            "--queue-depth" => {
                queue_depth = args.next().and_then(|s| s.parse().ok());
            }
            "--min-size" => {
                min_size = args.next().and_then(|s| s.parse().ok());
            }
            "--max-size" => {
                max_size = args.next().and_then(|s| s.parse().ok());
            }
            "--scale" => {
                scale = args.next().and_then(|s| s.parse().ok());
            }
            "--stride" => {
                stride = args.next().and_then(|s| s.parse().ok());
            }
            "--nms" => {
                nms = args.next().and_then(|s| s.parse().ok());
            }
            "--min-neighbors" => {
                min_neighbors = args.next().and_then(|s| s.parse().ok());
            }
            "--min-score" => {
                min_score = args.next().and_then(|s| s.parse().ok());
            }
            "--only-with-face" => {
                only_with_face = true;
            }
            "--no-gpu" => {
                no_gpu = true;
            }
            "--no-equalize" => {
                no_equalize = true;
            }
            "--cnn" => {
                use_cnn = true;
            }
            "--cnn-weights" => {
                cnn_weights_path = args.next().map(PathBuf::from);
            }
            "--algo" => {
                algo = args.next().map(|s| s.to_ascii_lowercase());
            }
            other if !other.starts_with("--") && input.is_none() => {
                input = Some(other.to_string());
            }
            other => {
                eprintln!("unknown argument: {}", other);
                print_help();
                std::process::exit(2);
            }
        }
    }

    let input = match input {
        Some(s) => s,
        None => {
            print_help();
            std::process::exit(2);
        }
    };
    if input == "demo" {
        run_demo(out.as_deref());
        return;
    }
    let out = match out {
        Some(p) => p,
        None => {
            eprintln!("--out <DIR> is required (or run `rs-face demo`)");
            std::process::exit(2);
        }
    };

    // Resolve algorithm: --algo wins over --cnn for clarity.
    let algo_name = match algo {
        Some(s) => s,
        None => {
            if use_cnn {
                "cnn".to_string()
            } else {
                "haar".to_string()
            }
        }
    };
    // yunet / mtcnn / hog were placeholder detectors and have been removed;
    // scrfd / arcface names stay recognised so users get a precise fallback
    // message pointing at the ONNX feature-gated example rather than a typo hint.
    let known: &[&str] = &["haar", "cnn", "luminance", "scrfd", "arcface"];
    if !known.contains(&algo_name.as_str()) {
        let suggestion = did_you_mean(&algo_name, known);
        match suggestion {
            Some(guess) => eprintln!(
                "[rs-face] unknown algorithm '{}'; did you mean '{}'? (known: {})",
                algo_name,
                guess,
                known.join(", "),
            ),
            None => eprintln!(
                "[rs-face] unknown algorithm '{}'; known: {}",
                algo_name,
                known.join(", "),
            ),
        }
        eprintln!("[rs-face] run with --list-algos for the full description table");
        std::process::exit(2);
    }
    println!("[rs-face] algorithm: {}", algo_name);

    // Load cascade.
    let mut cascade = if let Some(p) = cascade_path {
        match Cascade::load(&p) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("failed to load cascade {}: {}", p.display(), e);
                std::process::exit(2);
            }
        }
    } else {
        bundled_frontalface_cascade()
    };
    if let Some(b) = std::env::var("RS_FACE_CASCADE_BIAS")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        cascade.stage_bias = b;
        eprintln!("[rs-face] stage_bias overridden to {}", b);
    }
    println!(
        "[rs-face] cascade: {} stages, {} features, window {}x{}",
        cascade.num_stages(),
        cascade.num_features(),
        cascade.window_w,
        cascade.window_h
    );

    // Debug: dump first 3 stages' weak features for sanity checking.
    if std::env::var("RS_FACE_DEBUG").is_ok() {
        for (i, st) in cascade.stages.iter().take(3).enumerate() {
            eprintln!(
                "[debug] stage {} threshold={:.4}, {} weak features",
                i,
                st.stage_threshold,
                st.weak_features.len()
            );
            for (j, w) in st.weak_features.iter().take(3).enumerate() {
                eprintln!(
                    "[debug]   weak {}: feat_idx={} thr={:.4} sign={} left={:.4} right={:.4}",
                    j, w.feature_index, w.threshold, w.sign, w.left_val, w.right_val
                );
                let feat = &cascade.features[w.feature_index as usize];
                eprintln!(
                    "[debug]     feature kind={:?} {}x{} rects={}",
                    feat.kind,
                    feat.width,
                    feat.height,
                    feat.rects.len()
                );
                for r in &feat.rects {
                    eprintln!(
                        "[debug]       rect {} {} {}x{} weight={}",
                        r.x, r.y, r.w, r.h, r.weight
                    );
                }
            }
        }
    }

    // Open source.
    let mut src = match source::open(&input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to open source '{}': {}", input, e);
            std::process::exit(2);
        }
    };
    if let Some(total) = src.total_hint() {
        println!("[rs-face] source: {} (≈{} frames)", input, total);
    } else {
        println!("[rs-face] source: {} (live)", input);
    }

    let mut cfg = PipelineConfig::default();
    if let Some(t) = threads {
        cfg.threads = t;
    }
    if let Some(q) = queue_depth {
        cfg.queue_depth = q;
    }
    if let Some(v) = min_size {
        cfg.detector.min_size = v;
    }
    if let Some(v) = max_size {
        cfg.detector.max_size = v;
    }
    if let Some(v) = scale {
        cfg.detector.scale_factor = v;
    }
    if let Some(v) = stride {
        cfg.detector.window_stride = v;
    }
    if let Some(v) = nms {
        cfg.detector.nms_iou_threshold = v;
    }
    if let Some(v) = min_neighbors {
        cfg.detector.min_neighbors = v;
    }
    if let Some(v) = min_score {
        cfg.min_score = v;
    }
    cfg.only_with_face = only_with_face;
    cfg.detector.use_gpu = !no_gpu;
    cfg.detector.equalize_hist = !no_equalize;

    println!(
        "[rs-face] threads={}, queue_depth={}, detector={:?}",
        cfg.threads, cfg.queue_depth, cfg.detector
    );

    let t0 = Instant::now();
    let stats = match algo_name.as_str() {
        "haar" => match Pipeline::run(&mut *src, cascade, &out, cfg) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("pipeline error: {}", e);
                std::process::exit(1);
            }
        },
        "cnn" => match run_cnn_pipeline(&mut *src, &out, &cfg, cnn_weights_path.as_deref()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("cnn pipeline error: {}", e);
                std::process::exit(1);
            }
        },
        "luminance" => {
            match run_algo_pipeline(&mut *src, &out, &cfg, |img: &GrayImage| {
                LuminanceFaceDetector::new(LuminanceConfig::default()).detect(img)
            }) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("luminance pipeline error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        "scrfd" | "arcface" => {
            eprintln!(
                "--algo {} needs an ONNX feature (ort-backend / tract-backend); \
                 see examples/detect_scrfd_arcface.rs for the supported entry point",
                algo_name
            );
            std::process::exit(2);
        }
        other => {
            eprintln!("unknown --algo: {} (use haar|cnn|luminance)", other);
            std::process::exit(2);
        }
    };
    let wall_ms = t0.elapsed().as_millis() as u64;
    let fps = if wall_ms > 0 {
        stats.frames_processed as f32 * 1000.0 / wall_ms as f32
    } else {
        0.0
    };
    println!(
        "[rs-face] done: {} frames ({} with face), {} detections, wall {:.2}s, throughput {:.2} fps, detect avg {:.2} ms/frame",
        stats.frames_processed, stats.frames_with_face, stats.total_detections,
        wall_ms as f32 / 1000.0, fps, stats.detect_ms_avg
    );
    println!("[rs-face] output: {}/", out.display());
}

/// Zero-argument install check: run the bundled OpenCV frontal-face cascade
/// against a 256×256 grayscale portrait embedded in the binary. Prints the
/// detections, writes an annotated PNG, and exits non-zero if no face was
/// found so `rs-face demo` doubles as a post-install smoke test — no model
/// download, no sample files, zero third-party runtime.
fn run_demo(out_dir: Option<&std::path::Path>) {
    let out_dir = out_dir.unwrap_or_else(|| std::path::Path::new("rsface-demo"));
    let mut pgm: &[u8] = include_bytes!("../assets/demo_face_256.pgm");
    let gray = match rsface::image::codec::read_pgm(&mut pgm) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("[demo] embedded portrait decode failed: {e}");
            std::process::exit(1);
        }
    };
    let cascade = bundled_frontalface_cascade();
    println!(
        "[demo] bundled 256x256 portrait + bundled OpenCV frontalface cascade ({} stages, {} features)",
        cascade.num_stages(),
        cascade.num_features(),
    );
    // CPU deliberately: a smoke test must not dlopen/JIT a GPU stack.
    let det = Detector::new(
        cascade,
        DetectorConfig {
            use_gpu: false,
            ..DetectorConfig::default()
        },
    );
    let t0 = Instant::now();
    let hits = det.detect(&gray);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("[demo] {} face(s) in {:.1} ms:", hits.len(), ms);
    for h in &hits {
        println!(
            "       x={} y={} w={} h={} score={:.3}",
            h.x, h.y, h.w, h.h, h.score
        );
    }
    if hits.is_empty() {
        eprintln!("[demo] FAIL: the bundled cascade found no face in the bundled portrait");
        std::process::exit(1);
    }

    // Gray → RGB replication for the annotated writer.
    let (w, h) = (gray.width(), gray.height());
    let mut rgb = rsface::image::RgbImage::new(w, h);
    for y in 0..h {
        let row = rgb.row_mut(y);
        for (x, &v) in gray.row(y).iter().enumerate() {
            row[x * 3] = v;
            row[x * 3 + 1] = v;
            row[x * 3 + 2] = v;
        }
    }
    let rec = rsface::output::DetectionRecord {
        frame_index: 0,
        timestamp_ms: 0,
        image_file: String::new(),
        width: w,
        height: h,
        detections: hits
            .iter()
            .map(|d| rsface::Detection {
                x: d.x,
                y: d.y,
                w: d.w,
                h: d.h,
                score: d.score,
            })
            .collect(),
        detect_ms: ms,
    };
    let fname = match std::fs::create_dir_all(out_dir)
        .and_then(|_| rsface::output::write_annotated_png(out_dir, &rec, &rgb))
    {
        Ok(fname) => fname,
        Err(e) => {
            eprintln!(
                "[demo] failed to write annotated PNG under {}: {e}",
                out_dir.display()
            );
            std::process::exit(1);
        }
    };
    println!(
        "[demo] OK — annotated image written to {}/{}",
        out_dir.display(),
        fname
    );
}

/// CNN-only pipeline: runs the modern CNN detector on each frame, bypassing
/// the Viola-Jones pipeline entirely. This demonstrates that the project
/// ships a CNN face detector (the most-used modern algorithm family), even
/// though the weights are hand-crafted rather than pretrained.
fn run_cnn_pipeline(
    src: &mut dyn rsface::source::FrameSource,
    out_dir: &std::path::Path,
    cfg: &rsface::pipeline::PipelineConfig,
    weights_path: Option<&std::path::Path>,
) -> std::io::Result<rsface::pipeline::PipelineStats> {
    use rsface::cnn::{CnnConfig, CnnDetector, CnnWeights};
    use rsface::output::PipelineSummary;
    use rsface::pipeline::PipelineStats;

    std::fs::create_dir_all(out_dir)?;
    let cfg_cnn = CnnConfig {
        window_w: 24,
        window_h: 24,
        stride: cfg.detector.window_stride,
        confidence_threshold: cfg.min_score.max(0.5),
        max_size: cfg.detector.max_size,
    };
    let det = match weights_path {
        Some(p) => match CnnWeights::load(p) {
            Ok(w) => CnnDetector::with_weights(w, cfg_cnn),
            Err(e) => {
                eprintln!("failed to load CNN weights {}: {}", p.display(), e);
                std::process::exit(2);
            }
        },
        None => CnnDetector::new(cfg_cnn),
    };

    let start = std::time::Instant::now();
    let mut records = Vec::<rsface::output::DetectionRecord>::new();
    let mut frames_with_face: u64 = 0;
    let mut total_detections: u64 = 0;

    loop {
        let frame_opt = src.next_frame()?;
        let Some(frame) = frame_opt else {
            break;
        };
        let w = frame.gray.width();
        let h = frame.gray.height();
        // Convert u8 grayscale → f32 in [0, 1] for the CNN.
        let mut f32_img = vec![0.0f32; w * h];
        for (i, &p) in frame.gray.as_slice().iter().enumerate() {
            f32_img[i] = p as f32 / 255.0;
        }
        let dets = det.detect(&f32_img, w, h);
        let n = dets.len() as u64;
        if n > 0 {
            frames_with_face += 1;
        }
        total_detections += n;

        // Build RGB representation: prefer the source's RGB, fall back to
        // gray-to-RGB replication (one byte per channel, same value).
        let rgb = if let Some(arc) = &frame.rgb {
            (**arc).clone()
        } else {
            let mut rgb = rsface::image::RgbImage::new(w, h);
            for y in 0..h {
                let row = rgb.row_mut(y);
                let gray_row = frame.gray.row(y);
                for (x, &v) in gray_row.iter().enumerate() {
                    row[x * 3] = v;
                    row[x * 3 + 1] = v;
                    row[x * 3 + 2] = v;
                }
            }
            rgb
        };
        let rec = rsface::output::DetectionRecord {
            frame_index: frame.index,
            timestamp_ms: frame.timestamp_ms,
            image_file: String::new(),
            width: w,
            height: h,
            detections: dets
                .iter()
                .map(|d| rsface::Detection {
                    x: d.x,
                    y: d.y,
                    w: d.w,
                    h: d.h,
                    score: d.confidence,
                })
                .collect(),
            detect_ms: 0.0,
        };
        let fname = rsface::output::write_annotated_png(out_dir, &rec, &rgb)?;
        let mut rec = rec;
        rec.image_file = fname;
        if cfg.only_with_face && rec.detections.is_empty() {
            // Don't accumulate empty frames in the manifest when --only-with-face
            // is set (mirrors the Viola-Jones pipeline behaviour).
        } else {
            records.push(rec);
        }
    }
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let processed = records.len() as u64;
    rsface::output::write_manifest(
        &out_dir.join("manifest.json"),
        &records,
        &PipelineSummary {
            frames_processed: processed,
            frames_with_face,
            total_detections,
            elapsed_ms,
            detect_ms_total: 0.0,
        },
    )?;
    Ok(PipelineStats {
        frames_processed: processed,
        frames_with_face,
        total_detections,
        elapsed_ms,
        detect_ms_avg: 0.0,
    })
}

/// Generic pipeline for closure-supplied detectors (currently the
/// luminance heuristic). Handles frame I/O, RGB fallback, annotated PNG
/// write, and manifest generation; output matches the haar/cnn paths.
fn run_algo_pipeline<F>(
    src: &mut dyn rsface::source::FrameSource,
    out_dir: &std::path::Path,
    cfg: &rsface::pipeline::PipelineConfig,
    detect_fn: F,
) -> std::io::Result<rsface::pipeline::PipelineStats>
where
    F: Fn(&GrayImage) -> Vec<rsface::Detection>,
{
    use rsface::output::PipelineSummary;
    use rsface::pipeline::PipelineStats;

    std::fs::create_dir_all(out_dir)?;
    let start = std::time::Instant::now();
    let mut records = Vec::<rsface::output::DetectionRecord>::new();
    let mut frames_with_face: u64 = 0;
    let mut total_detections: u64 = 0;

    loop {
        let frame_opt = src.next_frame()?;
        let Some(frame) = frame_opt else {
            break;
        };
        let w = frame.gray.width();
        let h = frame.gray.height();
        let dets = detect_fn(&frame.gray);
        let n = dets.len() as u64;
        if n > 0 {
            frames_with_face += 1;
        }
        total_detections += n;

        let rgb = if let Some(arc) = &frame.rgb {
            (**arc).clone()
        } else {
            let mut rgb = rsface::image::RgbImage::new(w, h);
            for y in 0..h {
                let row = rgb.row_mut(y);
                let gray_row = frame.gray.row(y);
                for (x, &v) in gray_row.iter().enumerate() {
                    row[x * 3] = v;
                    row[x * 3 + 1] = v;
                    row[x * 3 + 2] = v;
                }
            }
            rgb
        };
        let rec = rsface::output::DetectionRecord {
            frame_index: frame.index,
            timestamp_ms: frame.timestamp_ms,
            image_file: String::new(),
            width: w,
            height: h,
            detections: dets,
            detect_ms: 0.0,
        };
        let fname = rsface::output::write_annotated_png(out_dir, &rec, &rgb)?;
        let mut rec = rec;
        rec.image_file = fname;
        if cfg.only_with_face && rec.detections.is_empty() {
        } else {
            records.push(rec);
        }
    }
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let processed = records.len() as u64;
    rsface::output::write_manifest(
        &out_dir.join("manifest.json"),
        &records,
        &PipelineSummary {
            frames_processed: processed,
            frames_with_face,
            total_detections,
            elapsed_ms,
            detect_ms_total: 0.0,
        },
    )?;
    Ok(PipelineStats {
        frames_processed: processed,
        frames_with_face,
        total_detections,
        elapsed_ms,
        detect_ms_avg: 0.0,
    })
}
