//! Golden-set evaluation: precision / recall / F1 + latency per
//! algorithm against a fixed ground-truth label set.
//!
//! This is the accuracy baseline the crate's "ensemble and
//! calibration" subagent uses to compare detectors and verify the
//! fusion module improves over any single algorithm.
//!
//! Run with:
//!   cargo test --test golden_eval -- --nocapture
//!
//! Labels live under `tests/fixtures/golden/labels/<image>.txt` (one
//! `<x> <y> <w> <h>` per line, pixel space, top-left origin). The set
//! is intentionally small — four images — so a calibration cycle stays
//! fast enough to iterate in seconds, not minutes.

use std::path::Path;
use std::time::Instant;

use rsface::cnn::{CnnConfig, CnnDetector};
use rsface::detector::{Detector, DetectorConfig};
use rsface::ensemble::{fuse, EnsembleConfig, FusedCluster, TaggedDetection};
use rsface::face::Detection;
use rsface::face_detector::FaceDetector;
use rsface::haar::bundled::bundled_frontalface_cascade;
use rsface::image::{codec, GrayImage, RgbImage};
use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

#[derive(Clone, Debug)]
struct GroundTruth {
    boxes: Vec<(usize, usize, usize, usize)>,
}

fn load_ground_truth(name: &str) -> GroundTruth {
    let path = Path::new("tests/fixtures/golden/labels").join(format!("{name}.txt"));
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing golden label {}: {e}. Run the inspect_algos test \
             or look at tests/fixtures/golden/ to bootstrap.",
            path.display()
        )
    });
    let mut boxes = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 4 {
            panic!(
                "label {}: line {} expected '<x> <y> <w> <h>', got {:?}",
                name,
                i + 1,
                parts
            );
        }
        let x: usize = parts[0].parse().expect("x");
        let y: usize = parts[1].parse().expect("y");
        let w: usize = parts[2].parse().expect("w");
        let h: usize = parts[3].parse().expect("h");
        boxes.push((x, y, w, h));
    }
    GroundTruth { boxes }
}

fn load_image(path: &str) -> GrayImage {
    let p = Path::new(path);
    let mut f = std::fs::File::open(p).expect("open");
    let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
    match ext {
        "pgm" => codec::read_pgm(&mut f).expect("read_pgm"),
        "ppm" => {
            let rgb: RgbImage = codec::read_ppm(&mut f).expect("read_ppm");
            rgb.to_gray()
        }
        _ => panic!("unsupported ext {ext}"),
    }
}

#[derive(Clone, Debug)]
struct ScoreReport {
    name: &'static str,
    precision: f32,
    recall: f32,
    f1: f32,
    /// Average detection time per image (milliseconds). The CNN's
    /// coarse stride keeps it under a second even in debug builds.
    avg_ms: f32,
    /// Total raw detections across the golden set (after any
    /// per-algorithm threshold). Includes over-fires.
    emitted_total: usize,
}

/// Greedy IoU matching: detections sorted by score, each matches the
/// highest-IoU unclaimed ground-truth box above the threshold. Returns
/// `(tp, fp)`. Standard PASCAL VOC / COCO convention.
fn match_detections(dets: &[Detection], gt: &GroundTruth, iou_threshold: f32) -> (usize, usize) {
    let mut sorted: Vec<&Detection> = dets.iter().collect();
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut claimed = vec![false; gt.boxes.len()];
    let mut tp = 0usize;
    let mut fp = 0usize;
    for det in sorted {
        let mut best: Option<usize> = None;
        let mut best_iou = iou_threshold;
        for (i, (gx, gy, gw, gh)) in gt.boxes.iter().enumerate() {
            if claimed[i] {
                continue;
            }
            let g = Detection {
                x: *gx,
                y: *gy,
                w: *gw,
                h: *gh,
                score: 1.0,
            };
            let cur = det.iou(&g);
            if cur > best_iou {
                best_iou = cur;
                best = Some(i);
            }
        }
        if let Some(i) = best {
            claimed[i] = true;
            tp += 1;
        } else {
            fp += 1;
        }
    }
    (tp, fp)
}

/// Run an algorithm over every image, accumulate counts and timing,
/// and compute the precision / recall / F1 against the labels.
///
/// `detect` is called once per image (so per-image measurements
/// include the load step implicitly — i.e. everything from the
/// already-decoded `GrayImage` to the `Vec<Detection>`). When the
/// entry's path is `None`, the `img_fn` closure produces the image
/// instead of loading it from disk — used by the synthetic-face
/// entry that is generated in-memory.
fn run_algo<F>(name: &'static str, imgs: &[ImageEntry], mut detect: F) -> ScoreReport
where
    F: FnMut(&GrayImage) -> Vec<Detection>,
{
    let mut total_ms = 0.0f32;
    let mut emitted_total = 0usize;
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut total_gt = 0usize;
    for entry in imgs {
        let img = load_entry(entry);
        let t0 = Instant::now();
        let dets = detect(&img);
        total_ms += t0.elapsed().as_secs_f32() * 1000.0;
        emitted_total += dets.len();
        let (t, f) = match_detections(&dets, &entry.gt, 0.5);
        tp += t;
        fp += f;
        total_gt += entry.gt.boxes.len();
    }
    let precision = if tp + fp > 0 {
        tp as f32 / (tp + fp) as f32
    } else {
        0.0
    };
    let recall = if total_gt > 0 {
        tp as f32 / total_gt as f32
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let avg_ms = total_ms / imgs.len() as f32;
    ScoreReport {
        name,
        precision,
        recall,
        f1,
        avg_ms,
        emitted_total,
    }
}

/// Same as `run_algo` but skips images larger than `max_pixels` — used
/// to keep the CNN's slow pure-Rust forward pass tractable on
/// biden-class images. Skipped entries still contribute their GT box
/// count to the recall denominator.
fn run_algo_skip_large<F>(
    name: &'static str,
    imgs: &[ImageEntry],
    max_pixels: usize,
    mut detect: F,
) -> ScoreReport
where
    F: FnMut(&GrayImage) -> Vec<Detection>,
{
    let mut total_ms = 0.0f32;
    let mut emitted_total = 0usize;
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut total_gt = 0usize;
    for entry in imgs {
        let img = load_entry(entry);
        if img.width() * img.height() > max_pixels {
            total_gt += entry.gt.boxes.len();
            continue;
        }
        let t0 = Instant::now();
        let dets = detect(&img);
        total_ms += t0.elapsed().as_secs_f32() * 1000.0;
        emitted_total += dets.len();
        let (t, f) = match_detections(&dets, &entry.gt, 0.5);
        tp += t;
        fp += f;
        total_gt += entry.gt.boxes.len();
    }
    let precision = if tp + fp > 0 {
        tp as f32 / (tp + fp) as f32
    } else {
        0.0
    };
    let recall = if total_gt > 0 {
        tp as f32 / total_gt as f32
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let avg_ms = total_ms / imgs.len() as f32;
    ScoreReport {
        name,
        precision,
        recall,
        f1,
        avg_ms,
        emitted_total,
    }
}

/// Synthetic-entry discriminator: maps each `path: None` entry to the
/// generator function that produces its image. Real fixtures go through
/// `load_image` instead. The variants below give us the 12-entry golden
/// set's full mix of real photos, profile-like patterns, noise patterns,
/// and a multi-face scene without requiring external image fixtures.
#[derive(Clone, Copy, Debug)]
enum SyntheticKind {
    /// The canonical "Haar fails, band-signature fires" 200x200 pattern.
    FrontalTuned,
    /// 240x240 profile-like pattern, box at (80, 80, 80, 80).
    ProfileA,
    /// 240x240 profile-like pattern, box at (50, 90, 100, 80).
    ProfileB,
    /// 240x240 profile-like pattern, box at (70, 60, 90, 100).
    ProfileC,
    /// 240x240 uniform noise, no face (tests variance-norm floor).
    NoiseUniform,
    /// 240x240 checkerboard, no face (tests high-FP regimes).
    NoiseChecker,
    /// 240x240 gradient, no face (tests low-variance windows).
    NoiseGradient,
    /// 320x240 with 3 small face-like patches left-to-right.
    MultiFace3,
}

struct ImageEntry {
    path: Option<&'static str>,
    gt: GroundTruth,
    /// `Some(kind)` for synthetic in-memory entries; `None` for fixtures
    /// loaded from disk.
    synthetic: Option<SyntheticKind>,
}

/// Load (or generate) the image for an entry. Real fixtures (with
/// `path: Some(_)`) are read from disk; synthetic entries are rebuilt
/// in-memory from their kind tag. The hardcoded GT box positions in
/// `golden_set` are the contract these generators must respect — a
/// rebuild that moves the box desynchronises labels from images.
fn load_entry(entry: &ImageEntry) -> GrayImage {
    match (entry.path, entry.synthetic) {
        (Some(p), _) => load_image(p),
        (None, Some(SyntheticKind::FrontalTuned)) => synthetic_face().0,
        (None, Some(SyntheticKind::ProfileA)) => synthetic_profile(80, 80, 80, 80),
        (None, Some(SyntheticKind::ProfileB)) => synthetic_profile(50, 90, 100, 80),
        (None, Some(SyntheticKind::ProfileC)) => synthetic_profile(70, 60, 90, 100),
        (None, Some(SyntheticKind::NoiseUniform)) => synthetic_noise_uniform(),
        (None, Some(SyntheticKind::NoiseChecker)) => synthetic_noise_checker(),
        (None, Some(SyntheticKind::NoiseGradient)) => synthetic_noise_gradient(),
        (None, Some(SyntheticKind::MultiFace3)) => synthetic_multi_face_3(),
        (None, None) => {
            // Default to the canonical frontal face for any legacy
            // entry that doesn't tag its synthetic kind.
            synthetic_face().0
        }
    }
}

/// Debug helper: print every detection each algorithm produces on
/// the golden set, alongside the GT box, so a human can verify the
/// labels are sensible and a misconfigured detector's failure mode is
/// obvious from the output. Not in the table; run on demand.
#[test]
#[ignore]
fn golden_eval_inspect() {
    let imgs = golden_set();
    let haar_det = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let lum = LuminanceFaceDetector::new(LuminanceConfig {
        stride: 8,
        ..LuminanceConfig::default()
    });
    let cnn_cal = CnnDetector::new(CnnConfig {
        stride: 16,
        max_size: 64,
        confidence_threshold: 0.99,
        ..CnnConfig::default()
    });

    for entry in &imgs {
        let img = load_entry(entry);
        let label = entry.path.unwrap_or("<synthetic>");
        println!(
            "\n== {} ({}x{}) GT={:?}",
            label,
            img.width(),
            img.height(),
            entry.gt.boxes
        );
        let haar_dets = haar_det.detect(&img);
        println!("  haar: {:?}", haar_dets);
        let lum_dets = lum.detect(&img);
        println!("  luminance: {:?}", lum_dets);
        if img.width() * img.height() <= 512 * 512 {
            let (w, h) = (img.width(), img.height());
            let mut buf = vec![0.0f32; w * h];
            for (i, &p) in img.as_slice().iter().enumerate() {
                buf[i] = p as f32 / 255.0;
            }
            let cnn_dets = cnn_cal.detect(&buf, w, h);
            println!(
                "  cnn-cal: {} detections (showing up to 10)",
                cnn_dets.len()
            );
            for d in cnn_dets.iter().take(10) {
                println!(
                    "    x={} y={} w={} h={} conf={:.3}",
                    d.x, d.y, d.w, d.h, d.confidence
                );
            }
        }
    }
}

fn golden_set() -> Vec<ImageEntry> {
    // Twelve-entry golden set (2026-09-21 expansion):
    //  Real frontal (4):
    //   - single frontal (lena)
    //   - two-person (two-people)
    //   - tiny embedded portrait (demo_face_256)
    //   - large editorial portrait (biden)
    //  Synthetic face-like (1):
    //   - in-memory synthetic face where Haar is known to miss; the
    //     canonical "Haar fails, band-signature detector fires" case
    //     where the ensemble has to add value.
    //  Synthetic profile-like (3):
    //   - profile variants that exercise tilted rect paths and the
    //     diagonal-edge feature family. None of them is a perfect
    //     profile silhouette, but they all look like a "face edge" and
    //     are calibrated to put the cascade at the boundary of
    //     acceptance so the precision/recall frontier is visible.
    //  Synthetic noise patterns (3):
    //   - uniform noise (zero-variance windows — variance-norm floor
    //     must keep these out), checkerboard (high-frequency texture
    //     confuses LBP-style detectors), and gradient (smooth gradient
    //     tests the variance pre-filter on low-variance scenes).
    //  Synthetic multi-face scene (1):
    //   - 3 small face-like patterns tiled into one 320x240 frame so
    //     group_rectangles has to actually group, not just dedupe.
    //
    // The biden.ppm 970×2204 entry is the slowest of the lot — every
    // algorithm's per-image cost is dominated by it (Haar ~6 s, CNN
    // ~minutes in debug). It is included to keep the per-algo F1
    // comparable across the "real photo" axis (small/medium/large).
    let mut entries = Vec::new();
    for (img, label) in &[
        ("tests/fixtures/lena.ppm", "lena"),
        ("tests/fixtures/two-people.ppm", "two-people"),
        ("tests/fixtures/demo_face_256.pgm", "demo_face_256"),
        ("tests/fixtures/biden.ppm", "biden"),
    ] {
        entries.push(ImageEntry {
            path: Some(img),
            gt: load_ground_truth(label),
            synthetic: None,
        });
    }

    // Synthetic face: in-memory; single GT box at (75, 75, 50, 50).
    entries.push(ImageEntry {
        path: None,
        gt: GroundTruth {
            boxes: vec![(75, 75, 50, 50)],
        },
        synthetic: Some(SyntheticKind::FrontalTuned),
    });

    // Synthetic profile variants (in-memory): each is a 240x240 pattern
    // whose strong horizontal gradient approximates a profile silhouette
    // with a bright forehead, dark eye band, and a sloped jaw. The
    // rectangular bounding boxes are intentionally loose — the goal is
    // to measure "did the cascade fire on the profile-like region",
    // not to evaluate sub-pixel box tightness.
    entries.push(ImageEntry {
        path: None,
        gt: GroundTruth {
            boxes: vec![(80, 80, 80, 80)],
        },
        synthetic: Some(SyntheticKind::ProfileA),
    });
    entries.push(ImageEntry {
        path: None,
        gt: GroundTruth {
            boxes: vec![(50, 90, 100, 80)],
        },
        synthetic: Some(SyntheticKind::ProfileB),
    });
    entries.push(ImageEntry {
        path: None,
        gt: GroundTruth {
            boxes: vec![(70, 60, 90, 100)],
        },
        synthetic: Some(SyntheticKind::ProfileC),
    });

    // Synthetic noise patterns (in-memory): each is a 240x240 image
    // with no face-like structure. The ground truth is EMPTY so a
    // detector's only valid response is "no detections" — any output
    // is a false positive and contributes to FP count.
    entries.push(ImageEntry {
        path: None,
        gt: GroundTruth { boxes: vec![] },
        synthetic: Some(SyntheticKind::NoiseUniform),
    });
    entries.push(ImageEntry {
        path: None,
        gt: GroundTruth { boxes: vec![] },
        synthetic: Some(SyntheticKind::NoiseChecker),
    });
    entries.push(ImageEntry {
        path: None,
        gt: GroundTruth { boxes: vec![] },
        synthetic: Some(SyntheticKind::NoiseGradient),
    });

    // Synthetic multi-face scene (in-memory): 320x240 frame with 3
    // small face-like patches tiled left-to-right. Group_rectangles
    // has to actually group the cascade's many similar hits per patch
    // into 3 distinct clusters, not just dedupe the entire image.
    entries.push(ImageEntry {
        path: None,
        gt: GroundTruth {
            boxes: vec![(20, 80, 70, 70), (130, 80, 70, 70), (240, 80, 70, 70)],
        },
        synthetic: Some(SyntheticKind::MultiFace3),
    });

    entries
}

/// Generate a synthetic 200×200 face pattern in memory: bright
/// forehead (top 40%), dark eye band (mid 30%), mid chin (bottom 30%),
/// plus edge spots at the eyes and nose. Returns a `GrayImage` plus
/// a single ground-truth box at (75, 75, 50, 50).
///
/// The pattern is intentionally tuned for the luminance detector's
/// band gate: forehead ≈ 200, mid ≈ 40 (well below the 90 cap), bot
/// ≈ 130, edge density high — so `score_window` returns Some(_) for
/// the central window. The Haar cascade may or may not fire on it;
/// the test exercises "what does the ensemble do when detectors
/// disagree on a synthetic face?".
#[allow(dead_code)]
fn synthetic_face() -> (GrayImage, (usize, usize, usize, usize)) {
    let mut img = GrayImage::new(200, 200);
    for y in 0..80 {
        for x in 0..200 {
            let v = 200 + (((x * 7 + y * 11) ^ (x >> 3)) & 0xF) as i32 - 7;
            img[(x, y)] = v.clamp(160, 230) as u8;
        }
    }
    for y in 80..140 {
        for x in 0..200 {
            let mut v = 40;
            if (60..90).contains(&x) || (110..140).contains(&x) {
                v = 12;
            }
            if (95..105).contains(&x) && y > 90 && y < 130 {
                v += 60;
            }
            img[(x, y)] = v;
        }
    }
    for y in 140..200 {
        for x in 0..200 {
            let v = 130 + (((x * 5 + y * 3) ^ (y >> 4)) & 0x7) as i32 - 3;
            img[(x, y)] = v.clamp(100, 160) as u8;
        }
    }
    (img, (75, 75, 50, 50))
}

/// Synthetic 240x240 "profile-like" pattern: a bright forehead, dark eye
/// band, and bright chin, but with the dark band offset to one side to
/// mimic the asymmetric shadow of a profile face. The `(bx, by, bw, bh)`
/// argument is the bounding box around the pattern's face region; the
/// image's edge geometry and intensity levels are calibrated so the
/// cascade's tilted rects and centre-surround features have non-zero
/// response at that window position.
///
/// These are *not* real profile faces — they are deliberately
/// under-detected to put the cascade at the boundary of acceptance so
/// the precision/recall frontier is visible. Real profiles would require
/// 1000s of annotated training samples and a full OpenCV-trained
/// `haarcascade_profileface.xml` cascade, which is out of scope for the
/// zero-dep library.
#[allow(dead_code)]
fn synthetic_profile(bx: usize, by: usize, bw: usize, bh: usize) -> GrayImage {
    let (w, h) = (240usize, 240usize);
    let mut img = GrayImage::new(w, h);
    // Background: mid-grey with low-amplitude noise so the variance
    // pre-filter does not kill the inner normrect outright.
    for y in 0..h {
        for x in 0..w {
            let v = 80 + ((x.wrapping_mul(13) ^ y.wrapping_mul(7)) & 0x1F) as i32 - 15;
            img[(x, y)] = v.clamp(40, 120) as u8;
        }
    }
    // Forehead band: bright (~200), top of the box.
    let fy0 = by;
    let fy1 = by + bh / 4;
    // Eye band: dark (~30), with a horizontal offset to simulate a
    // profile silhouette's nose shadow on one side.
    let ey0 = by + bh / 3;
    let ey1 = by + (bh * 2) / 3;
    let offset_x = bx + bw / 4;
    // Chin band: mid (~140), bottom of the box.
    let cy0 = by + (bh * 3) / 4;
    let cy1 = by + bh;
    for y in fy0..fy1.min(h) {
        for x in bx..(bx + bw).min(w) {
            let v = 200 + ((x.wrapping_mul(5) ^ y.wrapping_mul(11)) & 0x7) as i32 - 3;
            img[(x, y)] = v.clamp(180, 220) as u8;
        }
    }
    for y in ey0..ey1.min(h) {
        for x in bx..(bx + bw).min(w) {
            // Slight horizontal ramp: the dark band is darker on one
            // side to mimic a profile silhouette.
            let ramp = ((x as i32 - bx as i32) * 60 / bw.max(1) as i32) - 30;
            let v = 30 + ramp;
            img[(x, y)] = v.clamp(0, 60) as u8;
            // Nose-bridge streak: a single bright column at the offset.
            if (offset_x..offset_x + 3).contains(&x) {
                img[(x, y)] = 220;
            }
        }
    }
    for y in cy0..cy1.min(h) {
        for x in bx..(bx + bw).min(w) {
            let v = 140 + ((x.wrapping_mul(7) ^ y.wrapping_mul(3)) & 0x7) as i32 - 3;
            img[(x, y)] = v.clamp(120, 160) as u8;
        }
    }
    img
}

/// 240x240 uniform noise (every pixel ≈ 128). Ground truth is empty;
/// the cascade must report zero detections here. A correct cascade eval
/// relies on the variance pre-filter (which we already pass at the
/// `passes_variance` level) to skip these windows. After the
/// `variance_norm` floor fix, even zero-variance windows return a bounded
/// factor (≤ ~1000) so the cascade can still score them — the
/// pre-filter is the only line of defence for this entry.
#[allow(dead_code)]
fn synthetic_noise_uniform() -> GrayImage {
    let (w, h) = (240usize, 240usize);
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img[(x, y)] = 128;
        }
    }
    img
}

/// 240x240 high-frequency checkerboard (16x16 cells of 0 / 255). Ground
/// truth is empty. The high local variance inflates feature responses
/// *without* any face-like structure; a correctly-tuned cascade rejects
/// these via the stage-threshold sum, not the variance pre-filter. This
/// entry exists to measure the false-positive rate on busy backgrounds.
#[allow(dead_code)]
fn synthetic_noise_checker() -> GrayImage {
    let (w, h) = (240usize, 240usize);
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img[(x, y)] = if ((x / 16) + (y / 16)) & 1 == 0 {
                0
            } else {
                255
            };
        }
    }
    img
}

/// 240x240 vertical gradient (top=20, bottom=220). Ground truth is
/// empty. Low-frequency gradients have moderate variance, so the
/// pre-filter passes but the cascade should still reject (no
/// face-like structure). This entry measures the cascade's tolerance
/// to wide illumination gradients.
#[allow(dead_code)]
fn synthetic_noise_gradient() -> GrayImage {
    let (w, h) = (240usize, 240usize);
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = 20 + (y * 200 / h) as i32;
            img[(x, y)] = v.clamp(0, 255) as u8;
        }
    }
    img
}

/// 320x240 frame with three 70x70 face-like patches tiled left-to-right
/// at y=80. Each patch is a downsized copy of `synthetic_face()` so the
/// cascade recognises each as a face. `group_rectangles` has to actually
/// group the cascade's many similar hits per patch into 3 distinct
/// clusters, not just dedupe the entire image. The GT boxes are
/// `(20, 80, 70, 70)`, `(130, 80, 70, 70)`, `(240, 80, 70, 70)`; the
/// third one is on the very right edge of the frame and must still
/// survive the cascade's image-bounds clamp.
#[allow(dead_code)]
fn synthetic_multi_face_3() -> GrayImage {
    let (w, h) = (320usize, 240usize);
    let mut img = GrayImage::new(w, h);
    // Dark grey background so the variance pre-filter rejects the
    // empty regions cleanly.
    for y in 0..h {
        for x in 0..w {
            img[(x, y)] = 30;
        }
    }
    for (ox, _oy) in &[(20usize, 80usize), (130usize, 80usize), (240usize, 80usize)] {
        // Build a 70x70 face-like patch in-place using the same logic
        // as `synthetic_face` (top bright, mid dark, bottom mid).
        for dy in 0..70 {
            for dx in 0..70 {
                let x = ox + dx;
                let y = 80 + dy;
                if x >= w || y >= h {
                    continue;
                }
                let v = if dy < 28 {
                    200 + ((dx.wrapping_mul(7) ^ dy.wrapping_mul(11)) & 0xF) as i32 - 7
                } else if dy < 50 {
                    let mut v = 40;
                    if (20..30).contains(&dx) || (38..46).contains(&dx) {
                        v = 12;
                    }
                    if (32..38).contains(&dx) && dy > 33 && dy < 47 {
                        v += 60;
                    }
                    v
                } else {
                    130 + ((dx.wrapping_mul(5) ^ dy.wrapping_mul(3)) & 0x7) as i32 - 3
                };
                img[(x, y)] = v.clamp(0, 255) as u8;
            }
        }
    }
    img
}

/// Pair every algorithm against the synthetic face and report which
/// detectors fired. Used to demonstrate the ensemble's job: pick up
/// the boxes that any single detector misses when the others fail.
#[test]
#[ignore]
fn synthetic_face_detector_diagnostic() {
    let (img, gt) = synthetic_face();
    let haar_det = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let lum = LuminanceFaceDetector::new(LuminanceConfig {
        stride: 4,
        ..LuminanceConfig::default()
    });
    let cnn_cal = CnnDetector::new(CnnConfig {
        stride: 16,
        max_size: 64,
        confidence_threshold: 0.99,
        ..CnnConfig::default()
    });

    println!(
        "synthetic_face: GT={:?}, img {}x{}",
        gt,
        img.width(),
        img.height()
    );
    let haar_dets = haar_det.detect(&img);
    println!("  haar ({}):", haar_dets.len());
    for d in &haar_dets {
        println!(
            "    x={} y={} w={} h={} score={:.3}",
            d.x, d.y, d.w, d.h, d.score
        );
    }
    let lum_dets = lum.detect(&img);
    println!("  luminance ({}):", lum_dets.len());
    for d in &lum_dets {
        println!(
            "    x={} y={} w={} h={} score={:.3}",
            d.x, d.y, d.w, d.h, d.score
        );
    }
    let (w, h) = (img.width(), img.height());
    let mut buf = vec![0.0f32; w * h];
    for (i, &p) in img.as_slice().iter().enumerate() {
        buf[i] = p as f32 / 255.0;
    }
    let cnn_dets = cnn_cal.detect(&buf, w, h);
    println!("  cnn-cal ({}):", cnn_dets.len());
    for d in cnn_dets.iter().take(5) {
        println!(
            "    x={} y={} w={} h={} conf={:.3}",
            d.x, d.y, d.w, d.h, d.confidence
        );
    }
    if cnn_dets.len() > 5 {
        println!("    ... +{} more", cnn_dets.len() - 5);
    }
}

/// Pretty-print the eval table. The format is intentionally compact
/// so a quick `--nocapture` run gives a single-screen comparison.
fn print_table(reports: &[ScoreReport]) {
    println!("\n=== rs-face golden eval ===");
    println!(
        "{:<14}  {:>9}  {:>9}  {:>9}  {:>9}  {:>8}",
        "algo", "precision", "recall", "F1", "avg_ms", "emit"
    );
    println!("{}", "-".repeat(72));
    for r in reports {
        println!(
            "{:<14}  {:>9.3}  {:>9.3}  {:>9.3}  {:>9.2}  {:>8}",
            r.name, r.precision, r.recall, r.f1, r.avg_ms, r.emitted_total
        );
    }
    println!();
}

/// Cached per-image detections from each algorithm. The golden set
/// runs Haar/Luminance/CNN once per image and stores the output here,
/// so multiple ensemble variants reuse the same vector instead of
/// re-running every detector (which would dominate the test's wall
/// clock on the larger fixtures).
#[derive(Default)]
struct CachedDetections {
    haar: (Vec<Vec<Detection>>, f32), // (detections, total elapsed ms)
    luminance: (Vec<Vec<Detection>>, f32),
    luminance_strict: (Vec<Vec<Detection>>, f32),
    cnn_raw: (Vec<Vec<Detection>>, f32),
    cnn_cal: (Vec<Vec<Detection>>, f32),
}

impl CachedDetections {
    fn run(imgs: &[ImageEntry]) -> Self {
        let haar_det = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
        let lum = LuminanceFaceDetector::new(LuminanceConfig {
            stride: 8,
            ..LuminanceConfig::default()
        });
        let lum_strict = LuminanceFaceDetector::new(LuminanceConfig {
            stride: 8,
            score_threshold: 0.70,
            ..LuminanceConfig::default()
        });
        let cnn_raw = CnnDetector::new(CnnConfig {
            stride: 16,
            max_size: 64,
            confidence_threshold: 0.5,
            ..CnnConfig::default()
        });
        let cnn_cal = CnnDetector::new(CnnConfig {
            stride: 16,
            max_size: 64,
            confidence_threshold: 0.99,
            ..CnnConfig::default()
        });

        let mut haar_dets = Vec::with_capacity(imgs.len());
        let mut lum_dets = Vec::with_capacity(imgs.len());
        let mut lum_strict_dets = Vec::with_capacity(imgs.len());
        let mut raw = Vec::with_capacity(imgs.len());
        let mut cal = Vec::with_capacity(imgs.len());
        let mut haar_ms = 0.0f32;
        let mut lum_ms = 0.0f32;
        let mut lum_strict_ms = 0.0f32;
        let mut raw_ms = 0.0f32;
        let mut cal_ms = 0.0f32;
        for entry in imgs {
            let img = load_entry(entry);

            let t0 = Instant::now();
            haar_dets.push(haar_det.detect(&img));
            haar_ms += t0.elapsed().as_secs_f32() * 1000.0;

            let t0 = Instant::now();
            lum_dets.push(lum.detect(&img));
            lum_ms += t0.elapsed().as_secs_f32() * 1000.0;

            let t0 = Instant::now();
            lum_strict_dets.push(lum_strict.detect(&img));
            lum_strict_ms += t0.elapsed().as_secs_f32() * 1000.0;

            let (w, h) = (img.width(), img.height());
            let small_enough = w * h <= 512 * 512;
            if small_enough {
                let mut buf = vec![0.0f32; w * h];
                for (i, &p) in img.as_slice().iter().enumerate() {
                    buf[i] = p as f32 / 255.0;
                }
                let t0 = Instant::now();
                let r = cnn_raw.detect(&buf, w, h);
                raw_ms += t0.elapsed().as_secs_f32() * 1000.0;
                let t0 = Instant::now();
                let c = cnn_cal.detect(&buf, w, h);
                cal_ms += t0.elapsed().as_secs_f32() * 1000.0;
                raw.push(
                    r.into_iter()
                        .map(|d| Detection {
                            x: d.x,
                            y: d.y,
                            w: d.w,
                            h: d.h,
                            score: d.confidence,
                        })
                        .collect(),
                );
                cal.push(
                    c.into_iter()
                        .map(|d| Detection {
                            x: d.x,
                            y: d.y,
                            w: d.w,
                            h: d.h,
                            score: d.confidence,
                        })
                        .collect(),
                );
            } else {
                raw.push(Vec::new());
                cal.push(Vec::new());
            }
        }
        CachedDetections {
            haar: (haar_dets, haar_ms),
            luminance: (lum_dets, lum_ms),
            luminance_strict: (lum_strict_dets, lum_strict_ms),
            cnn_raw: (raw, raw_ms),
            cnn_cal: (cal, cal_ms),
        }
    }

    fn by_name<'a>(&'a self, idx: usize, name: &str) -> &'a [Detection] {
        match name {
            "haar" => &self.haar.0[idx],
            "luminance" => &self.luminance.0[idx],
            "luminance-strict" => &self.luminance_strict.0[idx],
            "cnn-raw" => &self.cnn_raw.0[idx],
            "cnn-cal" => &self.cnn_cal.0[idx],
            _ => &[],
        }
    }
}

/// Helper: run an ensemble pass over the cached per-image detections
/// with the given `tag_weights` and `min_votes`, and accumulate
/// per-cluster scores into a `ScoreReport`. The detector passes
/// themselves happened earlier in `CachedDetections::run`; this
/// function only pays the (cheap) `fuse` + IoU-matching cost.
fn run_ensemble(
    name: &'static str,
    imgs: &[ImageEntry],
    cache: &CachedDetections,
    tag_weights: &[(&'static str, f32)],
    min_votes: usize,
    extra_filter_haar: bool,
) -> ScoreReport {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut total_gt = 0usize;
    let mut total_ms = 0.0f32;
    let mut fused_total = 0usize;
    for (i, entry) in imgs.iter().enumerate() {
        let t0 = Instant::now();
        let mut inputs: Vec<TaggedDetection> = Vec::new();
        for (src, weight) in tag_weights {
            for d in cache.by_name(i, src) {
                inputs.push(TaggedDetection::new(d.clone(), src, *weight));
            }
        }

        let cfg = EnsembleConfig {
            iou_threshold: 0.3,
            min_votes,
            ..EnsembleConfig::default()
        };
        let mut fused: Vec<FusedCluster> = fuse(inputs, &cfg);
        if extra_filter_haar {
            fused.retain(|c| c.sources.split('+').any(|s| s == "haar"));
        }
        fused_total += fused.len();
        let dets: Vec<Detection> = fused.iter().map(|c| c.detection.clone()).collect();
        let (t, f) = match_detections(&dets, &entry.gt, 0.5);
        tp += t;
        fp += f;
        total_gt += entry.gt.boxes.len();
        total_ms += t0.elapsed().as_secs_f32() * 1000.0;
    }

    let precision = if tp + fp > 0 {
        tp as f32 / (tp + fp) as f32
    } else {
        0.0
    };
    let recall = if total_gt > 0 {
        tp as f32 / total_gt as f32
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let avg_ms = total_ms / imgs.len() as f32;
    ScoreReport {
        name,
        precision,
        recall,
        f1,
        avg_ms,
        emitted_total: fused_total,
    }
}

/// Main eval: each algorithm in turn, then several ensemble variants,
/// in a single side-by-side table.
///
/// Marked `#[ignore]` because it requires `tests/fixtures/golden/labels/*.txt`,
/// which are gitignored (annotated per-image). Bootstrap with:
///
/// ```text
/// cargo test --test inspect_algos -- --ignored --nocapture
/// ```
///
/// Then write one `tests/fixtures/golden/labels/<image>.txt` per image,
/// with one `x y w h` line per ground-truth face box (pixel coords).
///
/// Run this eval with:
/// ```text
/// cargo test --test golden_eval -- --ignored --nocapture --test-threads=1
/// ```
#[test]
#[ignore = "requires gitignored tests/fixtures/golden/labels/*.txt — bootstrap via inspect_algos"]
fn golden_eval_table() {
    let imgs = golden_set();
    assert!(
        !imgs.is_empty(),
        "golden set is empty — populate tests/fixtures/golden/labels/"
    );
    eprintln!(
        "[golden-eval] {} images, {} GT boxes total",
        imgs.len(),
        imgs.iter().map(|e| e.gt.boxes.len()).sum::<usize>()
    );

    // Run every algorithm once per image and stash the results. The
    // multiple ensemble variants downstream reuse these vectors
    // instead of re-running the detectors, which would dominate the
    // test wall-clock.
    eprintln!("[golden-eval] running per-image detectors (cached)...");
    let cache = CachedDetections::run(&imgs);
    eprintln!("[golden-eval] detector pass complete");

    // --- Per-algorithm reports from the cached results. ---
    let haar_report = report_from_cache("haar", &imgs, &cache.haar.0, cache.haar.1);
    eprintln!(
        "[golden-eval] haar: P={:.3} R={:.3} F1={:.3} emit={}",
        haar_report.precision, haar_report.recall, haar_report.f1, haar_report.emitted_total
    );
    let lum_report = report_from_cache("luminance", &imgs, &cache.luminance.0, cache.luminance.1);
    eprintln!(
        "[golden-eval] luminance: P={:.3} R={:.3} F1={:.3} emit={}",
        lum_report.precision, lum_report.recall, lum_report.f1, lum_report.emitted_total
    );
    let lum_strict_report = report_from_cache(
        "luminance-strict",
        &imgs,
        &cache.luminance_strict.0,
        cache.luminance_strict.1,
    );
    eprintln!(
        "[golden-eval] luminance-strict: P={:.3} R={:.3} F1={:.3} emit={}",
        lum_strict_report.precision,
        lum_strict_report.recall,
        lum_strict_report.f1,
        lum_strict_report.emitted_total
    );
    let cnn_raw_report = report_from_cache("cnn-raw", &imgs, &cache.cnn_raw.0, cache.cnn_raw.1);
    eprintln!(
        "[golden-eval] cnn-raw: emit={}",
        cnn_raw_report.emitted_total
    );
    let cnn_cal_report = report_from_cache("cnn-cal", &imgs, &cache.cnn_cal.0, cache.cnn_cal.1);
    eprintln!(
        "[golden-eval] cnn-cal: emit={}",
        cnn_cal_report.emitted_total
    );

    // --- Ensemble variants. Each call below is now cheap: just a
    // fuse() + IoU match per image. The detector passes are reused. ---

    // (a) union, min_votes = 1: every cluster becomes a detection;
    //     precision is dominated by the noisier detectors' FPs.
    let ens_union = run_ensemble(
        "ensemble-union",
        &imgs,
        &cache,
        &[("haar", 1.0), ("luminance-strict", 0.7), ("cnn-cal", 0.4)],
        1,
        false,
    );
    eprintln!(
        "[golden-eval] ensemble-union: P={:.3} R={:.3} F1={:.3}",
        ens_union.precision, ens_union.recall, ens_union.f1
    );

    // (b) consensus, min_votes = 2: at least two algorithms must
    //     agree on the same window; singleton hits are dropped.
    let ens_consensus = run_ensemble(
        "ensemble-consensus",
        &imgs,
        &cache,
        &[("haar", 1.0), ("luminance-strict", 0.7), ("cnn-cal", 0.4)],
        2,
        false,
    );
    eprintln!(
        "[golden-eval] ensemble-consensus: P={:.3} R={:.3} F1={:.3}",
        ens_consensus.precision, ens_consensus.recall, ens_consensus.f1
    );

    // (c) haar-only ensemble (the safe baseline): no other algorithm
    //     contributes, so the result is just the haar set rebadged.
    //     Useful as a control: it shows the cost of adding noise.
    let ens_haar_only = run_ensemble(
        "ensemble-haar-only",
        &imgs,
        &cache,
        &[("haar", 1.0)],
        1,
        false,
    );
    eprintln!(
        "[golden-eval] ensemble-haar-only: P={:.3} R={:.3} F1={:.3}",
        ens_haar_only.precision, ens_haar_only.recall, ens_haar_only.f1
    );

    // (d) haar-gated ensemble: Haar is the primary signal; clusters
    //     are kept only when at least one Haar member participates.
    //     Luminance + CNN can refine the box and the score, but they
    //     cannot add a brand-new detection on their own. This is the
    //     deployment default — it preserves Haar's measured precision
    //     while letting the geometry of agreeing detectors nudge the
    //     box tighter.
    let ens_haar_gated = run_ensemble(
        "ensemble-haar-gated",
        &imgs,
        &cache,
        &[("haar", 1.0), ("luminance-strict", 0.7), ("cnn-cal", 0.4)],
        1,
        true,
    );
    eprintln!(
        "[golden-eval] ensemble-haar-gated: P={:.3} R={:.3} F1={:.3}",
        ens_haar_gated.precision, ens_haar_gated.recall, ens_haar_gated.f1
    );

    let reports = vec![
        haar_report,
        lum_report,
        lum_strict_report,
        cnn_raw_report,
        cnn_cal_report,
        ens_union,
        ens_consensus,
        ens_haar_only,
        ens_haar_gated,
    ];

    print_table(&reports);
}

/// Build a `ScoreReport` from cached per-image detections. The total
/// wall-clock `ms_total` is divided by the image count to produce the
/// `avg_ms` column.
fn report_from_cache(
    name: &'static str,
    imgs: &[ImageEntry],
    detections: &[Vec<Detection>],
    ms_total: f32,
) -> ScoreReport {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut total_gt = 0usize;
    let mut emitted_total = 0usize;
    for (i, entry) in imgs.iter().enumerate() {
        emitted_total += detections[i].len();
        let (t, f) = match_detections(&detections[i], &entry.gt, 0.5);
        tp += t;
        fp += f;
        total_gt += entry.gt.boxes.len();
    }
    let precision = if tp + fp > 0 {
        tp as f32 / (tp + fp) as f32
    } else {
        0.0
    };
    let recall = if total_gt > 0 {
        tp as f32 / total_gt as f32
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let avg_ms = ms_total / imgs.len() as f32;
    ScoreReport {
        name,
        precision,
        recall,
        f1,
        avg_ms,
        emitted_total,
    }
}
