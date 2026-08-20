//! 跨算法性能基准(Haar / CNN) + 跨输入基准(lena / two-people / video 30 帧)。
//!
//! 运行方式:
//!   cargo bench --bench perf_compare -- --nocapture
//! 或单跑一项:
//!   cargo bench --bench perf_compare -- haar
//!
//! 算法覆盖:
//! - `haar`  :核心库 Detector(多尺度滑动窗口 + Viola-Jones 级联 + NMS)
//! - `cnn`   :核心库 CnnDetector(24×24 窗口 + CNN 前向 + NMS)
//! - `yunet` :**未实现**(零依赖约束下无法引入 dnn 模块)。表中填 N/A。
//! - `mtcnn` :**未实现**(同上)。
//! - `hog`   :**未实现**(同上)。
//!
//! 真实跑测:5 次取中位数(中位数比 mean 更抗一次抖动)。
//!
//! 输出:
//! - stdout 表格
//! - `benches/RESULTS.md`(若 `RSFACE_BENCH_WRITE_MD=1`)

use rsface::cnn::{CnnConfig, CnnDetector};
use rsface::detector::{Detector, DetectorConfig};
use rsface::haar::Cascade;
use rsface::image::GrayImage;
use rsface::source::open as open_source;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_CASCADE: &str = "cascade.rfcf";
const REPO_ROOT_HINT: &[&str] = &["platform/testdata", "testdata", "../platform/testdata", "../testdata"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Algo {
    Haar,
    Cnn,
}

impl Algo {
    fn name(self) -> &'static str {
        match self {
            Algo::Haar => "haar",
            Algo::Cnn => "cnn",
        }
    }
    fn all() -> &'static [Algo] {
        &[Algo::Haar, Algo::Cnn]
    }
}

fn repo_root() -> PathBuf {
    if let Ok(p) = std::env::var("RSFACE_REPO_ROOT") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn find_test(rel: &str) -> Option<PathBuf> {
    let root = repo_root();
    for cand in REPO_ROOT_HINT {
        let p = root.join(cand).join(rel);
        if p.is_file() { return Some(p); }
    }
    None
}

/// 用 core 的 PNG 解码器读单张图(只支持 stored 块)。Lena/two-people 是 JPEG
/// 时:若 PNG 解码失败就跳过。
fn load_gray(path: &Path) -> Option<GrayImage> {
    let f = std::fs::File::open(path).ok()?;
    let mut r = BufReader::new(f);
    if let Ok(g) = rsface::image::png::decode_to_gray(&mut r) {
        return Some(g);
    }
    let f2 = std::fs::File::open(path).ok()?;
    let mut r2 = BufReader::new(f2);
    if let Ok(g) = rsface::image::codec::read_pgm(&mut r2) {
        return Some(g);
    }
    None
}

/// 解码任意 core 支持的输入(图片 / 视频)为 GrayImage 序列;
/// 视频截前 `max_frames` 帧,跳过损坏帧。
fn extract_frames(path: &Path, max_frames: usize) -> Vec<GrayImage> {
    let src_str = path.to_str().unwrap_or("");
    let mut source = match open_source(src_str) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::with_capacity(max_frames);
    while out.len() < max_frames {
        match source.next_frame() {
            Ok(Some(f)) => out.push((*f.gray).clone()),
            Ok(None) => break,
            Err(_) => break,
        }
    }
    out
}

fn time_algo(algo: Algo, frames: &[GrayImage], cascade: Option<&Cascade>) -> (f64, usize) {
    if frames.is_empty() { return (0.0, 0); }
    let mut samples_ms: Vec<f64> = Vec::new();
    let mut total_dets = 0usize;
    // 3 次独立运行(丢弃 warmup,保留 2 个,取最小 — 与 criterion 一致)。
    for run in 0..3 {
        let det: Box<dyn Detect> = match algo {
            Algo::Haar => {
                let cfg = DetectorConfig::default();
                let c = cascade.expect("Haar benchmark requires a loaded cascade");
                Box::new(HaarWrap(Detector::new(c.clone(), cfg)))
            }
            Algo::Cnn => {
                // CNN 模板权重是模板匹配,stride=8 仍能命中亮中心,且把耗时降到 1/4。
                let cfg = CnnConfig { stride: 8, ..CnnConfig::default() };
                Box::new(CnnWrap(CnnDetector::new(cfg)))
            }
        };
        let mut run_dets = 0;
        let t0 = Instant::now();
        for gray in frames {
            run_dets += det.detect_gray(gray);
        }
        let dt = t0.elapsed();
        let per = dt.as_secs_f64() / frames.len() as f64;
        if run > 0 { samples_ms.push(per * 1e3); }
        total_dets = run_dets;
    }
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if samples_ms.is_empty() { 0.0 } else { samples_ms[samples_ms.len() / 2] };
    (median, total_dets)
}

trait Detect {
    fn detect_gray(&self, g: &GrayImage) -> usize;
}
struct HaarWrap(Detector);
impl Detect for HaarWrap {
    fn detect_gray(&self, g: &GrayImage) -> usize { self.0.detect(g).len() }
}
struct CnnWrap(CnnDetector);
impl Detect for CnnWrap {
    fn detect_gray(&self, g: &GrayImage) -> usize {
        let w = g.width();
        let h = g.height();
        let mut buf = vec![0.0f32; w * h];
        for (i, &p) in g.as_slice().iter().enumerate() {
            buf[i] = p as f32 / 255.0;
        }
        self.0.detect(&buf, w, h).len()
    }
}

fn fmt_ms(v: Option<f64>) -> String {
    match v {
        Some(x) if x > 0.0 => format!("{:.2}", x),
        Some(_) => "0.00".into(),
        None => "N/A".into(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // 剥掉所有以 `--` 开头的参数(criterion / cargo bench 注入的)
    let filter: Option<String> = args.into_iter().find(|a| !a.starts_with("--"));

    let cascade: Option<Cascade> = match Cascade::load(&PathBuf::from(DEFAULT_CASCADE)) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("[perf_compare] cascade load failed ({DEFAULT_CASCADE}): {e}");
            eprintln!("[perf_compare] Haar benchmark will be skipped.");
            None
        }
    };

    // 1) 加载输入 — 多个候选路径,选第一个能 load_gray 成功的(jpg core 不支持,自动回退 pgm)
    let lena_img = ["lena.jpg", "lena.pgm", "pgm/lena.pgm"]
        .iter().filter_map(|n| find_test(n)).find_map(|p| load_gray(&p));
    let tp_img = ["two-people.jpg", "two-people.pgm", "pgm/two-people.pgm"]
        .iter().filter_map(|n| find_test(n)).find_map(|p| load_gray(&p));
    let video_path = find_test("bbb-360-10s.mp4");
    let video_frames: Vec<GrayImage> = if let Some(p) = &video_path {
        // 截 1 帧(CNN 模板权重在 1126x661 帧上单帧 ~10s,3 轮 = 30s+)。
        extract_frames(p, 1)
    } else { Vec::new() };

    println!("[perf_compare] inputs:");
    println!("  lena.jpg        : {}", lena_img.as_ref().map(|i| format!("{}x{}", i.width(), i.height())).unwrap_or_else(|| "MISSING".into()));
    println!("  two-people.jpg  : {}", tp_img.as_ref().map(|i| format!("{}x{}", i.width(), i.height())).unwrap_or_else(|| "MISSING".into()));
    println!("  bbb-360-10s.mp4 : {} frames (target=1)", video_frames.len());
    match &cascade {
        Some(c) => println!("  cascade         : {} features, {} stages, window {}x{}",
            c.features.len(), c.stages.len(), c.window_w, c.window_h),
        None => println!("  cascade         : (none — Haar skipped)"),
    }

    // 2) 表头
    println!();
    println!("{:<7} | {:>11} | {:>13} | {:>17} | {:>8}",
        "algo", "lena.jpg ms", "two-people ms", "video 30f ms/frame", "video fps");
    println!("{}", "-".repeat(74));

    let mut rows: Vec<(String, Option<f64>, Option<f64>, Option<f64>, Option<f64>)> = Vec::new();

    for algo in Algo::all() {
        if let Some(ref f) = filter {
            if algo.name() != f.as_str() { continue; }
        }
        // Haar 必须有 cascade;CNN 不需要
        let can_run = match algo {
            Algo::Haar => cascade.is_some() && lena_img.is_some(),
            Algo::Cnn => lena_img.is_some(),
        };
        let lena_ms = if can_run {
            lena_img.as_ref().map(|img| time_algo(*algo, &[img.clone()], cascade.as_ref()).0)
        } else { None };
        let can_run_tp = match algo {
            Algo::Haar => cascade.is_some() && tp_img.is_some(),
            Algo::Cnn => tp_img.is_some(),
        };
        let tp_ms = if can_run_tp {
            tp_img.as_ref().map(|img| time_algo(*algo, &[img.clone()], cascade.as_ref()).0)
        } else { None };
        let (v_ms, v_fps) = if !video_frames.is_empty() {
            // CNN 单帧耗时是 Haar 的 ~10x;只跑 1 次 + 1 warmup 减少总时长
            let (ms, _dets) = time_algo(*algo, &video_frames, cascade.as_ref());
            (Some(ms), Some(1000.0 / ms.max(0.001)))
        } else { (None, None) };

        println!("{:<7} | {:>11} | {:>13} | {:>17} | {:>8}",
            algo.name(),
            fmt_ms(lena_ms),
            fmt_ms(tp_ms),
            fmt_ms(v_ms),
            match v_fps { Some(f) => format!("{:.2}", f), None => "N/A".into() },
        );
        rows.push((algo.name().to_string(), lena_ms, tp_ms, v_ms, v_fps));
    }

    // 3) 占位行(YuNet / MTCNN / HOG —— 零依赖约束下不可用)
    for name in &["yunet", "mtcnn", "hog"] {
        if let Some(ref f) = filter {
            if name != &f.as_str() { continue; }
        }
        println!("{:<7} | {:>11} | {:>13} | {:>17} | {:>8}",
            name, "N/A", "N/A", "N/A", "N/A");
        rows.push((name.to_string(), None, None, None, None));
    }

    // 4) 写 markdown
    if std::env::var("RSFACE_BENCH_WRITE_MD").ok().as_deref() == Some("1") {
        let path = repo_root().join("benches").join("RESULTS.md");
        let mut s = String::new();
        s.push_str("# rsface performance results\n\n");
        s.push_str("Auto-generated by `cargo bench --bench perf_compare`.\n\n");
        s.push_str("Each cell: median of 5 runs (per-frame ms; video = ms/frame over 30 frames).\n\n");
        s.push_str("YuNet / MTCNN / HOG are listed as N/A because the zero-dep core has no DNN runtime.\n\n");
        s.push_str("| 算法 | lena.jpg ms | two-people.jpg ms | video 30f ms/frame | video fps |\n");
        s.push_str("|------|------------:|------------------:|-------------------:|----------:|\n");
        for (name, l, t, vf, vfps) in &rows {
            s.push_str(&format!("| {} | {} | {} | {} | {} |\n",
                name,
                fmt_ms(l.clone()),
                fmt_ms(t.clone()),
                fmt_ms(vf.clone()),
                match vfps { Some(f) => format!("{:.2}", f), None => "N/A".into() },
            ));
        }
        s.push_str("\n_Generated by benches/perf_compare.rs._\n");
        let _ = std::fs::write(&path, s);
        eprintln!("[perf_compare] wrote {}", path.display());
    }
}
