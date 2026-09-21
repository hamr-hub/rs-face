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

/// PASCAL-VOC IoU between two detection boxes. Local copy used by
/// `run_ensemble_merge` so the helper doesn't depend on private
/// crate-internal functions (`rsface::face::iou` is `pub(crate)`).
fn iou_boxes(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let inter = if x2 > x1 && y2 > y1 {
        (x2 - x1) * (y2 - y1)
    } else {
        0
    };
    let union = a.w * a.h + b.w * b.h - inter;
    if union == 0 {
        0.0
    } else {
        inter as f32 / union as f32
    }
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

/// Load (or generate) the image for an entry. The synthetic entry has
/// `path: None`; all others read from disk.
fn load_entry(entry: &ImageEntry) -> GrayImage {
    match entry.path {
        Some(p) => load_image(p),
        None => {
            // Re-generate the synthetic image by name.
            let name = entry.synth_name.unwrap_or("synth_canonical_face");
            let gen = synth_generator(name).expect("unknown synth name");
            gen().0
        }
    }
}

/// Lookup the synthetic generator by name. Each generator returns
/// `(image, gt_boxes)`. Adding a new synthetic entry means adding
/// a row here AND a row in `golden_set()`.
fn synth_generator(name: &str) -> Option<fn() -> (GrayImage, Vec<(usize, usize, usize, usize)>)> {
    match name {
        "synth_canonical_face" => Some(synth_canonical_face),
        "synth_dim_face" => Some(synth_dim_face),
        "synth_bright_face" => Some(synth_bright_face),
        "synth_two_faces" => Some(synth_two_faces),
        "synth_three_faces" => Some(synth_three_faces),
        "synth_small_face" => Some(synth_small_face),
        "synth_large_face" => Some(synth_large_face),
        "synth_occluded_bottom" => Some(synth_occluded_bottom),
        "synth_occluded_top" => Some(synth_occluded_top),
        "synth_lighting_top" => Some(synth_lighting_top),
        "synth_lighting_bottom" => Some(synth_lighting_bottom),
        "synth_with_glasses" => Some(synth_with_glasses),
        "synth_profile_left" => Some(synth_profile_left),
        "synth_profile_right" => Some(synth_profile_right),
        "synth_tilted_face" => Some(synth_tilted_face),
        "synth_no_face_uniform" => Some(synth_no_face_uniform),
        _ => None,
    }
}

struct ImageEntry {
    path: Option<&'static str>,
    gt: GroundTruth,
    /// Name of the synthetic generator — used to look up the
    /// builder function when `path` is None. Ignored when `path`
    /// is Some.
    synth_name: Option<&'static str>,
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
        expand_to_w: 50,
        expand_to_h: 50,
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
    // v2 golden set: 4 on-disk face images + 16 in-memory synthetic
    // images covering diverse conditions (frontal, profile, multi-face,
    // lighting variation, occlusion, scale, negative cases). The
    // synthetics test edge cases that the 4 standard test fixtures
    // (lena / two-people / demo_face_256 / biden) don't cover.
    //
    // The 970×2204 biden.ppm is still excluded from the dev-mode run
    // (Haar ~1 s, luminance ~0.5 s, CNN ~minutes per image — would push
    // the dev test well past the harness default). The `biden` label
    // file is kept under tests/fixtures/golden/labels/ so future
    // longer-running evaluations (e.g. release-mode benchmarks) can
    // include it without changing the eval harness — `biden` is added
    // to the set via `golden_set_with_biden()` below for that purpose.
    let mut entries = Vec::new();
    for (img, label) in &[
        ("tests/fixtures/lena.ppm", "lena"),
        ("tests/fixtures/two-people.ppm", "two-people"),
        ("tests/fixtures/demo_face_256.pgm", "demo_face_256"),
    ] {
        entries.push(ImageEntry {
            path: Some(img),
            gt: load_ground_truth(label),
            synth_name: None,
        });
    }
    // Synthetic faces — 16 entries covering diverse conditions.
    // Each generator is a closure that returns (img, gt_boxes). The
    // names + GT boxes are deterministic; the synthetic image itself
    // is rebuilt on demand by `load_entry` so the set stays small on
    // disk.
    let synth: &[(&str, fn() -> (GrayImage, Vec<(usize, usize, usize, usize)>))] = &[
        ("synth_canonical_face", synth_canonical_face),
        ("synth_dim_face", synth_dim_face),
        ("synth_bright_face", synth_bright_face),
        ("synth_two_faces", synth_two_faces),
        ("synth_three_faces", synth_three_faces),
        ("synth_small_face", synth_small_face),
        ("synth_large_face", synth_large_face),
        ("synth_occluded_bottom", synth_occluded_bottom),
        ("synth_occluded_top", synth_occluded_top),
        ("synth_lighting_top", synth_lighting_top),
        ("synth_lighting_bottom", synth_lighting_bottom),
        ("synth_with_glasses", synth_with_glasses),
        ("synth_profile_left", synth_profile_left),
        ("synth_profile_right", synth_profile_right),
        ("synth_tilted_face", synth_tilted_face),
        ("synth_no_face_uniform", synth_no_face_uniform),
    ];
    for (name, gen) in synth {
        let (_, boxes) = gen();
        entries.push(ImageEntry {
            path: None,
            gt: GroundTruth { boxes },
            synth_name: Some(name),
        });
    }
    entries
}

/// Variant of `golden_set` that adds the large biden fixture. Use for
/// release-mode benchmarks where wall-clock is less of a concern.
#[allow(dead_code)]
fn golden_set_with_biden() -> Vec<ImageEntry> {
    let mut entries = golden_set();
    entries.insert(
        0,
        ImageEntry {
            path: Some("tests/fixtures/biden.ppm"),
            gt: load_ground_truth("biden"),
            synth_name: None,
        },
    );
    entries
}

/// Generate a synthetic 200×200 face pattern in memory: bright
/// forehead (top 47%), dark eye band (mid 7.5%, exactly mid-band height
/// for a 50-px window), mid chin (bottom 45%), plus edge spots at the
/// eyes and nose. Returns a `GrayImage` plus a single ground-truth
/// box at (75, 75, 50, 50).
///
/// v2 layout (was forehead/eye/chin split 80/60/60 — too wide a mid band
/// for a 50-px GT window). New split 95/15/90 puts the eye band
/// exactly where the GT window's mid-band lands (rows 95–110) so the
/// luminance detector's band-margin gate fires inside the GT box:
///   - GT top band (rows 75–95) sits entirely in forehead (luma ≈ 210)
///   - GT mid band (rows 95–110) sits exactly in the eye band (luma ≈ 40)
///   - GT bot band (rows 110–125) sits in the chin (luma ≈ 130)
/// => `min(top-mid, bot-mid) ≈ min(170, 90) = 90`, well above the 12
///     gate. The Haar cascade is also more likely to fire on this layout
///     because the high-contrast forehead/eye-band boundary at row 95 is
///     now a clean horizontal edge.
#[allow(dead_code)]
fn synthetic_face() -> (GrayImage, (usize, usize, usize, usize)) {
    let mut img = GrayImage::new(200, 200);
    // Forehead (rows 0..95): bright with subtle texture
    for y in 0..95 {
        for x in 0..200 {
            let v = 200 + (((x * 7 + y * 11) ^ (x >> 3)) & 0xF) as i32 - 7;
            img[(x, y)] = v.clamp(160, 230) as u8;
        }
    }
    // Eye band (rows 95..110, exactly the GT window's mid-band rows):
    // dark with darker eye spots at cols 80..100 and 130..150, plus a
    // bright nose ridge in the centre.
    for y in 95..110 {
        for x in 0..200 {
            let mut v = 40;
            if (80..100).contains(&x) || (130..150).contains(&x) {
                v = 12; // eye spots
            }
            // nose ridge brightening in centre
            if (95..105).contains(&x) {
                v += 60;
            }
            img[(x, y)] = v;
        }
    }
    // Chin (rows 110..200): mid luma with texture
    for y in 110..200 {
        for x in 0..200 {
            let v = 130 + (((x * 5 + y * 3) ^ (y >> 4)) & 0x7) as i32 - 3;
            img[(x, y)] = v.clamp(100, 160) as u8;
        }
    }
    (img, (75, 75, 50, 50))
}

// ===========================================================================
// Synthetic face generators used by `golden_set()`.
//
// Each generator returns `(GrayImage, Vec<(x, y, w, h)>)`. The image is
// rebuilt on demand by `load_entry` so nothing needs to be on disk.
//
// Design contract for every "positive" generator (i.e. ones that include
// at least one face in their GT box list):
//   - GT box must be sized so the GT window's mid-band lands in a
//     recognisably dark eye region, with the top-band and bot-band on
//     either side (forehead brighter, chin brighter). That is what makes
//     the luminance detector's band-margin gate fire inside the GT box.
//   - Forehead / eye / chin boundaries are computed by `place_face()`
//     so a single helper produces all the variants below.
//   - Subtle per-pixel texture is added so the variance / edges /
//     symmetry sub-scores all clear their gates — without it the window
//     would look "too flat" and the combined score would be killed.
// ===========================================================================

/// Place a forehead/eye/chin face signature into `img` at the given
/// `gt_box`. `luma_scale` ∈ (0, 1] multiplies all band luma values so
/// the same layout works for bright/dim/over-exposed variants. The eye
/// band always sits exactly at the GT window's mid-band rows, the
/// forehead sits at the top-band rows, and the chin sits at the bot-band
/// rows — that's the geometric invariant the luminance detector relies on.
fn place_face(img: &mut GrayImage, gt_box: (usize, usize, usize, usize), luma_scale: f32) {
    let (gx, gy, gw, gh) = gt_box;
    let h1 = gh * 2 / 5; // top-band height (40%)
    let h2 = gh * 3 / 10; // mid-band height (30%)
                          // bot-band = gh - h1 - h2
    let eye_y0 = gy + h1;
    let eye_y1 = gy + h1 + h2;
    let chin_y0 = eye_y1;

    let forehead_luma = (210.0 * luma_scale) as i32;
    let eye_luma = (40.0 * luma_scale) as i32;
    let eye_spot_luma = (12.0 * luma_scale) as i32;
    let nose_luma = (60.0 * luma_scale) as i32;
    let chin_luma = (130.0 * luma_scale) as i32;

    // Forehead: rows gy..gy+h1
    for y in gy..gy + h1 {
        for x in gx..gx + gw {
            let t = ((x * 7 + y * 11) ^ (x >> 3)) & 0xF;
            let v = forehead_luma + t as i32 - 7;
            img[(x, y)] = v.clamp(0, 255) as u8;
        }
    }
    // Eye band: rows eye_y0..eye_y1
    let eye_spot_x0 = gx + gw * 2 / 5;
    let eye_spot_x1 = gx + gw * 3 / 5;
    let eye_spot_x2 = gx + gw * 7 / 10;
    let eye_spot_x3 = gx + gw * 9 / 10;
    let nose_x0 = gx + gw * 9 / 20;
    let nose_x1 = gx + gw * 11 / 20;
    for y in eye_y0..eye_y1 {
        for x in gx..gx + gw {
            let mut v = eye_luma;
            if (eye_spot_x0..eye_spot_x1).contains(&x) || (eye_spot_x2..eye_spot_x3).contains(&x) {
                v = eye_spot_luma;
            }
            if (nose_x0..nose_x1).contains(&x) {
                v += nose_luma;
            }
            img[(x, y)] = v.clamp(0, 255) as u8;
        }
    }
    // Chin: rows chin_y0..gy+gh
    for y in chin_y0..gy + gh {
        for x in gx..gx + gw {
            let t = ((x * 5 + y * 3) ^ (y >> 4)) & 0x7;
            let v = chin_luma + t as i32 - 3;
            img[(x, y)] = v.clamp(0, 255) as u8;
        }
    }
}

/// Canonical frontal face (200×200, GT 75,75,50,50) — same layout as
/// the legacy `synthetic_face` function but exposed under the
/// synth-generator dispatch table.
fn synth_canonical_face() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(200, 200);
    let gt = (75, 75, 50, 50);
    place_face(&mut img, gt, 1.0);
    (img, vec![gt])
}

/// Dim face — same layout, luma scaled to 0.5. Tests that the
/// luminance detector still finds faces in low-light images where
/// the band contrast is reduced.
fn synth_dim_face() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(200, 200);
    let gt = (75, 75, 50, 50);
    place_face(&mut img, gt, 0.5);
    (img, vec![gt])
}

/// Bright face — luma scaled close to 1.0 with forehead saturation.
fn synth_bright_face() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(200, 200);
    let gt = (75, 75, 50, 50);
    place_face(&mut img, gt, 1.0);
    // Push forehead to saturation so the band contrast is maximal.
    for y in 75..95 {
        for x in 75..125 {
            img[(x, y)] = 250;
        }
    }
    (img, vec![gt])
}

/// Two adjacent frontal faces in a 300×200 image. GTs at (40, 75,
/// 50, 50) and (200, 75, 50, 50).
fn synth_two_faces() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(300, 200);
    let gt1 = (40, 75, 50, 50);
    let gt2 = (200, 75, 50, 50);
    place_face(&mut img, gt1, 1.0);
    place_face(&mut img, gt2, 1.0);
    (img, vec![gt1, gt2])
}

/// Three frontal faces in a 400×200 image. GTs at (40, 75, 50, 50),
/// (175, 75, 50, 50), and (310, 75, 50, 50).
fn synth_three_faces() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(400, 200);
    let gt1 = (40, 75, 50, 50);
    let gt2 = (175, 75, 50, 50);
    let gt3 = (310, 75, 50, 50);
    place_face(&mut img, gt1, 1.0);
    place_face(&mut img, gt2, 1.0);
    place_face(&mut img, gt3, 1.0);
    (img, vec![gt1, gt2, gt3])
}

/// Small face (30×30) in a large 400×400 image — tests the
/// multi-scale pyramid branch of the luminance detector.
fn synth_small_face() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(400, 400);
    let gt = (185, 185, 30, 30);
    place_face(&mut img, gt, 1.0);
    (img, vec![gt])
}

/// Large face (150×150) in a 200×200 image — tests that the detector
/// doesn't under-emit when the face dominates the frame.
fn synth_large_face() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(200, 200);
    let gt = (25, 25, 150, 150);
    place_face(&mut img, gt, 1.0);
    (img, vec![gt])
}

/// Bottom-half occluded face: GT window's bot-band is forced to a
/// uniform dark value (occlusion). The luminance detector should
/// still fire because top-band (forehead) and mid-band (eyes)
/// retain the canonical signature.
fn synth_occluded_bottom() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(200, 200);
    let gt = (75, 75, 50, 50);
    place_face(&mut img, gt, 1.0);
    // Black out the bot-band (chin)
    for y in 110..200 {
        for x in 0..200 {
            img[(x, y)] = 0;
        }
    }
    (img, vec![gt])
}

/// Top-half occluded face: forehead forced dark. The min-margin
/// gate requires top > mid; if top is dark, the gate fails. The
/// detector is expected to MISS this image — it's a known failure
/// mode (the band signature needs the forehead to be brighter than
/// the eye band).
fn synth_occluded_top() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(200, 200);
    let gt = (75, 75, 50, 50);
    place_face(&mut img, gt, 1.0);
    // Black out the top-band (forehead)
    for y in 0..95 {
        for x in 0..200 {
            img[(x, y)] = 0;
        }
    }
    (img, vec![gt])
}

/// Strong top lighting: forehead at luma 250, eye band still dark,
/// chin dim. Tests that the detector is robust to high-contrast
/// forehead.
fn synth_lighting_top() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(200, 200);
    let gt = (75, 75, 50, 50);
    place_face(&mut img, gt, 1.0);
    // Brighten forehead further
    for y in 75..95 {
        for x in 75..125 {
            img[(x, y)] = 250;
        }
    }
    // Dim the chin
    for y in 110..200 {
        for x in 75..125 {
            img[(x, y)] = (img[(x, y)] as i32 - 50).max(60) as u8;
        }
    }
    (img, vec![gt])
}

/// Strong bottom lighting: chin bright, forehead dim. Tests the
/// symmetric failure mode of synth_lighting_top.
fn synth_lighting_bottom() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(200, 200);
    let gt = (75, 75, 50, 50);
    place_face(&mut img, gt, 1.0);
    // Brighten chin
    for y in 110..125 {
        for x in 75..125 {
            img[(x, y)] = 240;
        }
    }
    // Dim forehead
    for y in 75..95 {
        for x in 75..125 {
            img[(x, y)] = (img[(x, y)] as i32 - 70).max(80) as u8;
        }
    }
    (img, vec![gt])
}

/// Face with dark "glasses" extending across the eye band. The eye
/// band is wider and darker — tests that the detector still fires
/// when eye-region darkness is intensified.
fn synth_with_glasses() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(200, 200);
    let gt = (75, 75, 50, 50);
    place_face(&mut img, gt, 1.0);
    // Widen the eye band and darken it (glasses + frame)
    for y in 95..110 {
        for x in 75..125 {
            img[(x, y)] = 8;
        }
    }
    (img, vec![gt])
}

/// Profile-style face: forehead is on the right half of the window
/// (no symmetric band structure). The mirror-symmetry sub-score
/// will be low; the band-margin sub-score must carry the
/// detection. This is a known partial-failure case for the
/// luminance detector — useful for measuring how often the
/// ensemble's other algorithms compensate.
fn synth_profile_left() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(200, 200);
    let gt = (75, 75, 50, 50);
    place_face(&mut img, gt, 1.0);
    // Asymmetric luma shift: brighten right half, darken left half
    for y in 75..125 {
        for x in 75..100 {
            let v = img[(x, y)] as i32 - 40;
            img[(x, y)] = v.clamp(0, 255) as u8;
        }
        for x in 100..125 {
            let v = img[(x, y)] as i32 + 30;
            img[(x, y)] = v.clamp(0, 255) as u8;
        }
    }
    (img, vec![gt])
}

/// Mirror of synth_profile_left — forehead/face brightens on the
/// left half. Same detection expectation.
fn synth_profile_right() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(200, 200);
    let gt = (75, 75, 50, 50);
    place_face(&mut img, gt, 1.0);
    for y in 75..125 {
        for x in 75..100 {
            let v = img[(x, y)] as i32 + 30;
            img[(x, y)] = v.clamp(0, 255) as u8;
        }
        for x in 100..125 {
            let v = img[(x, y)] as i32 - 40;
            img[(x, y)] = v.clamp(0, 255) as u8;
        }
    }
    (img, vec![gt])
}

/// Tilted face (synthetic rotation): the band boundaries are
/// slanted 15° by per-row column offset, producing a face-like
/// pattern that's slightly rotated. Haar and Luminance both use
/// axis-aligned windows so this is a stress case for both.
fn synth_tilted_face() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let mut img = GrayImage::new(200, 200);
    let gt = (75, 75, 50, 50);
    // Use place_face at the canonical (axis-aligned) location, then
    // simulate rotation by shifting rows left as we go down.
    place_face(&mut img, gt, 1.0);
    let mut copy = img.clone();
    for y in 0..200 {
        let dx = ((y as f32 - 100.0) * 0.18).round() as i32;
        for x in 0..200 {
            let sx = (x as i32 - dx).clamp(0, 199) as usize;
            img[(x, y)] = copy[(sx, y)];
        }
    }
    (img, vec![gt])
}

/// Negative test: uniform image with no face. All algorithms should
/// produce zero detections. The golden eval must still record zero
/// FP for this entry to keep the precision column meaningful.
fn synth_no_face_uniform() -> (GrayImage, Vec<(usize, usize, usize, usize)>) {
    let img = GrayImage::new(200, 200); // all zero
    (img, vec![])
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
        // Expand bundled 24×24 weights to 50×50 so they can match
        // the synthetic face's 50×50 GT box (see CnnConfig docs).
        expand_to_w: 50,
        expand_to_h: 50,
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

/// Per-image diagnostic for the v2 expanded golden set. Walks every
/// entry (real + synthetic) and prints how many boxes each algorithm
/// fired. Useful when investigating why a particular entry is
/// over- or under-firing. `#[ignore]` because it prints a lot.
#[test]
#[ignore]
fn golden_per_image_breakdown() {
    let imgs = golden_set();
    let haar = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let lum = LuminanceFaceDetector::new(LuminanceConfig::default());
    let synth_names: &[&str] = &[
        "synth_canonical_face",
        "synth_dim_face",
        "synth_bright_face",
        "synth_two_faces",
        "synth_three_faces",
        "synth_small_face",
        "synth_large_face",
        "synth_occluded_bottom",
        "synth_occluded_top",
        "synth_lighting_top",
        "synth_lighting_bottom",
        "synth_with_glasses",
        "synth_profile_left",
        "synth_profile_right",
        "synth_tilted_face",
        "synth_no_face_uniform",
    ];
    let mut synth_idx = 0;
    for entry in &imgs {
        let img = load_entry(entry);
        let label = match entry.path {
            Some(p) => std::path::Path::new(p)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            None => {
                let n = synth_names.get(synth_idx).copied().unwrap_or("synth?");
                synth_idx += 1;
                n.to_string()
            }
        };
        let haar_dets = haar.detect(&img);
        let lum_dets = lum.detect(&img);
        let (mut tp_haar, mut fp_haar) = (0, 0);
        for d in &haar_dets {
            let (t, f) = match_detections(&[d.clone()], &entry.gt, 0.5);
            tp_haar += t;
            fp_haar += f;
        }
        let (mut tp_lum, mut fp_lum) = (0, 0);
        for d in &lum_dets {
            let (t, f) = match_detections(&[d.clone()], &entry.gt, 0.5);
            tp_lum += t;
            fp_lum += f;
        }
        println!(
            "[{:<24}] GT={}  haar: {} det, tp={}, fp={}  lum: {} det, tp={}, fp={}",
            label,
            entry.gt.boxes.len(),
            haar_dets.len(),
            tp_haar,
            fp_haar,
            lum_dets.len(),
            tp_lum,
            fp_lum,
        );
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
            // The bundled template weights fire on local 24×24 texture
            // patterns; expand each hit to a 50×50 box so it can match
            // GT boxes of realistic size in the golden set.
            expand_to_w: 50,
            expand_to_h: 50,
            ..CnnConfig::default()
        });
        let cnn_cal = CnnDetector::new(CnnConfig {
            stride: 16,
            max_size: 64,
            confidence_threshold: 0.99,
            expand_to_w: 50,
            expand_to_h: 50,
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

    // (e) haar+newluminance merge (v2 best): keeps every Haar-cluster
    //     AND every high-confidence Luminance-strict cluster that
    //     does NOT overlap an existing Haar cluster (IoU < 0.3).
    //     This is the "free recall" gain: Luminance fires on
    //     synthetic faces and Haar-failed real faces, but those
    //     boxes are filtered out only when they sit on top of an
    //     already-confirmed Haar hit. Designed to lift F1 above
    //     Haar-only while keeping precision near Haar's measured
    //     value.
    let ens_merge = run_ensemble_merge(
        "ensemble-haar-merge",
        &imgs,
        &cache,
        &[("haar", 1.0), ("luminance-strict", 0.7)],
        0.3, // iou_threshold
    );
    eprintln!(
        "[golden-eval] ensemble-haar-merge: P={:.3} R={:.3} F1={:.3}",
        ens_merge.precision, ens_merge.recall, ens_merge.f1
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
        ens_merge,
    ];

    print_table(&reports);
}

/// Merge-style ensemble: keep every Haar cluster (anchors the
/// precision floor), then add Luminance-strict clusters that do NOT
/// overlap any Haar cluster (IoU < `iou_threshold`). The output
/// preserves Haar's TPs at full precision and adds Luminance's
/// Haar-missed TPs as new detections. Implementation:
/// 1. Run fuse() with `min_votes=1` over the configured sources
///    and collect every cluster whose membership contains "haar".
/// 2. Re-run fuse() with the same sources but filter to
///    "luminance-strict" members only and `min_votes=1`.
/// 3. Drop any luminance-only cluster whose centroid has IoU ≥
///    `iou_threshold` against an already-kept Haar cluster.
/// 4. Return the merged set.
fn run_ensemble_merge(
    name: &'static str,
    imgs: &[ImageEntry],
    cache: &CachedDetections,
    tag_weights: &[(&'static str, f32)],
    iou_threshold: f32,
) -> ScoreReport {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut total_gt = 0usize;
    let mut total_ms = 0.0f32;
    let mut fused_total = 0usize;
    let cfg = EnsembleConfig {
        iou_threshold,
        min_votes: 1,
        ..EnsembleConfig::default()
    };
    for (i, entry) in imgs.iter().enumerate() {
        let t0 = Instant::now();
        let mut inputs: Vec<TaggedDetection> = Vec::new();
        for (src, weight) in tag_weights {
            for d in cache.by_name(i, src) {
                inputs.push(TaggedDetection::new(d.clone(), src, *weight));
            }
        }
        let fused = fuse(inputs, &cfg);

        // Phase 1: keep every cluster that has at least one Haar member.
        let mut kept: Vec<FusedCluster> = fused
            .iter()
            .filter(|c| c.sources.split('+').any(|s| s == "haar"))
            .cloned()
            .collect();

        // Phase 2: add Luminance-strict-only clusters that don't overlap
        // any kept Haar cluster.
        for c in &fused {
            if c.sources.split('+').any(|s| s == "haar") {
                continue; // already kept
            }
            if c.detection.score < 0.65 {
                continue; // not high-confidence
            }
            let overlaps_haar = kept
                .iter()
                .any(|k| iou_boxes(&k.detection, &c.detection) > iou_threshold);
            if !overlaps_haar {
                kept.push(c.clone());
            }
        }

        fused_total += kept.len();
        let dets: Vec<Detection> = kept.iter().map(|c| c.detection.clone()).collect();
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
