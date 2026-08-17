//! rs-face CLI — face detection on a video URL or image sequence.

use std::path::PathBuf;
use std::time::Instant;

use rsface::haar::Cascade;
use rsface::pipeline::{Pipeline, PipelineConfig};
use rsface::source;

fn print_help() {
    println!(
        "rs-face — zero-dep Viola-Jones face detector\n\n\
         USAGE:\n  \
         rs-face <INPUT> --out <DIR> [options]\n\n\
         INPUT forms:\n  \
           test://N            synthetic test pattern (N frames)\n  \
           /path/to/dir        image sequence (PNG/PPM/JPG files)\n  \
           /path/file.png|jpg  single image\n  \
           http(s)://host/p    single image or PNG sequence base URL\n  \
           *.mp4|*.mov|*.avi|*.mkv|*.webm | rtsp://...\n                           (requires `ffmpeg` on PATH)\n\n\
         OPTIONS:\n  \
           --out <DIR>           output directory (required)\n  \
           --cascade <PATH>      load cascade from .rfcf file (default: built-in demo)\n  \
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
           --help                print this help\n"
    );
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
    let mut min_score: Option<f32> = None;
    let mut only_with_face = false;
    let mut no_gpu = false;
    let mut use_cnn = false;
    let mut cnn_weights_path: Option<PathBuf> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--help" | "-h" => { print_help(); return; }
            "--out" => { out = args.next().map(PathBuf::from); }
            "--cascade" => { cascade_path = args.next().map(PathBuf::from); }
            "--threads" => { threads = args.next().and_then(|s| s.parse().ok()); }
            "--queue-depth" => { queue_depth = args.next().and_then(|s| s.parse().ok()); }
            "--min-size" => { min_size = args.next().and_then(|s| s.parse().ok()); }
            "--max-size" => { max_size = args.next().and_then(|s| s.parse().ok()); }
            "--scale" => { scale = args.next().and_then(|s| s.parse().ok()); }
            "--stride" => { stride = args.next().and_then(|s| s.parse().ok()); }
            "--nms" => { nms = args.next().and_then(|s| s.parse().ok()); }
            "--min-score" => { min_score = args.next().and_then(|s| s.parse().ok()); }
            "--only-with-face" => { only_with_face = true; }
            "--no-gpu" => { no_gpu = true; }
            "--cnn" => { use_cnn = true; }
            "--cnn-weights" => { cnn_weights_path = args.next().map(PathBuf::from); }
            other if !other.starts_with("--") && input.is_none() => { input = Some(other.to_string()); }
            other => {
                eprintln!("unknown argument: {}", other);
                print_help();
                std::process::exit(2);
            }
        }
    }

    let input = match input {
        Some(s) => s,
        None => { print_help(); std::process::exit(2); }
    };
    let out = match out {
        Some(p) => p,
        None => { eprintln!("--out <DIR> is required"); std::process::exit(2); }
    };

    // Load cascade.
    let cascade = if let Some(p) = cascade_path {
        match Cascade::load(&p) {
            Ok(c) => c,
            Err(e) => { eprintln!("failed to load cascade {}: {}", p.display(), e); std::process::exit(2); }
        }
    } else {
        rsface::haar::params::demo_face_cascade()
    };
    println!("[rs-face] cascade: {} stages, {} features, window {}x{}",
             cascade.num_stages(), cascade.num_features(), cascade.window_w, cascade.window_h);

    // Debug: dump first 3 stages' weak features for sanity checking.
    if std::env::var("RS_FACE_DEBUG").is_ok() {
        for (i, st) in cascade.stages.iter().take(3).enumerate() {
            eprintln!("[debug] stage {} threshold={:.4}, {} weak features", i, st.stage_threshold, st.weak_features.len());
            for (j, w) in st.weak_features.iter().take(3).enumerate() {
                eprintln!("[debug]   weak {}: feat_idx={} thr={:.4} sign={} left={:.4} right={:.4}",
                    j, w.feature_index, w.threshold, w.sign, w.left_val, w.right_val);
                let feat = &cascade.features[w.feature_index as usize];
                eprintln!("[debug]     feature kind={:?} {}x{} rects={}",
                    feat.kind, feat.width, feat.height, feat.rects.len());
                for r in &feat.rects {
                    eprintln!("[debug]       rect {} {} {}x{} weight={}", r.x, r.y, r.w, r.h, r.weight);
                }
            }
        }
    }

    // Open source.
    let mut src = match source::open(&input) {
        Ok(s) => s,
        Err(e) => { eprintln!("failed to open source '{}': {}", input, e); std::process::exit(2); }
    };
    if let Some(total) = src.total_hint() {
        println!("[rs-face] source: {} (≈{} frames)", input, total);
    } else {
        println!("[rs-face] source: {} (live)", input);
    }

    let mut cfg = PipelineConfig::default();
    if let Some(t) = threads { cfg.threads = t; }
    if let Some(q) = queue_depth { cfg.queue_depth = q; }
    if let Some(v) = min_size { cfg.detector.min_size = v; }
    if let Some(v) = max_size { cfg.detector.max_size = v; }
    if let Some(v) = scale { cfg.detector.scale_factor = v; }
    if let Some(v) = stride { cfg.detector.window_stride = v; }
    if let Some(v) = nms { cfg.detector.nms_iou_threshold = v; }
    if let Some(v) = min_score { cfg.min_score = v; }
    cfg.only_with_face = only_with_face;
    cfg.detector.use_gpu = !no_gpu;

    println!("[rs-face] threads={}, queue_depth={}, detector={:?}", cfg.threads, cfg.queue_depth, cfg.detector);

    let t0 = Instant::now();
    let stats = if use_cnn {
        // CNN detector path: bypass the Viola-Jones pipeline and run the
        // modern CNN forward pass directly on each frame.
        let stats = match run_cnn_pipeline(&mut *src, &out, &cfg, cnn_weights_path.as_deref()) {
            Ok(s) => s,
            Err(e) => { eprintln!("cnn pipeline error: {}", e); std::process::exit(1); }
        };
        stats
    } else {
        match Pipeline::run(&mut *src, cascade, &out, cfg) {
            Ok(s) => s,
            Err(e) => { eprintln!("pipeline error: {}", e); std::process::exit(1); }
        }
    };
    let wall_ms = t0.elapsed().as_millis() as u64;
    let fps = if wall_ms > 0 { stats.frames_processed as f32 * 1000.0 / wall_ms as f32 } else { 0.0 };
    println!(
        "[rs-face] done: {} frames ({} with face), {} detections, wall {:.2}s, throughput {:.2} fps",
        stats.frames_processed, stats.frames_with_face, stats.total_detections,
        wall_ms as f32 / 1000.0, fps
    );
    println!("[rs-face] output: {}/", out.display());
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
    use rsface::pipeline::PipelineStats;
    use rsface::output::PipelineSummary;

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
        let Some(frame) = frame_opt else { break; };
        let w = frame.gray.width();
        let h = frame.gray.height();
        // Convert u8 grayscale → f32 in [0, 1] for the CNN.
        let mut f32_img = vec![0.0f32; w * h];
        for (i, &p) in frame.gray.as_slice().iter().enumerate() {
            f32_img[i] = p as f32 / 255.0;
        }
        let dets = det.detect(&f32_img, w, h);
        let n = dets.len() as u64;
        if n > 0 { frames_with_face += 1; }
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
                    row[x*3] = v; row[x*3+1] = v; row[x*3+2] = v;
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
            detections: dets.iter().map(|d| rsface::Detection {
                x: d.x, y: d.y, w: d.w, h: d.h, score: d.confidence,
            }).collect(),
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
        },
    )?;
    Ok(PipelineStats {
        frames_processed: processed,
        frames_with_face,
        total_detections,
        elapsed_ms,
    })
}
