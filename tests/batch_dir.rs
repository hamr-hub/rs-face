//! Integration tests for `rsface::batch::run_batch_dir`.
//!
//! These exercise the public end-to-end surface: feed the bundled demo
//! portrait (which the bundled Haar cascade detects a face in) into a temp
//! directory, run `run_batch_dir`, then verify the manifest + per-image
//! annotated PNGs and the `only_with-face` filter.
//!
//! Run with:
//!   `cargo test --test batch_dir`

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rsface::batch::{self, BatchConfig};
use rsface::haar::bundled::bundled_frontalface_cascade;

struct TempDir(PathBuf);

impl TempDir {
    fn new(prefix: &str) -> std::io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("{prefix}-{pid}-{n}"));
        fs::create_dir_all(&path)?;
        Ok(Self(path))
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixtures() -> &'static str {
    // CARGO_MANIFEST_DIR at runtime is set by cargo to the crate root.
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures")
}

#[test]
fn batch_dir_processes_real_face_fixture_and_writes_per_image_outputs() {
    let in_dir = TempDir::new("rsface-batch-in").unwrap();
    let out_dir = TempDir::new("rsface-batch-out").unwrap();
    // The bundled demo portrait is guaranteed to produce ≥1 detection with the
    // bundled OpenCV Haar cascade (see `rs-face demo`'s self-check).
    let src = PathBuf::from(fixtures()).join("demo_face_256.pgm");
    fs::copy(&src, in_dir.path().join("demo_face_256.pgm")).unwrap();

    let (stats, results) = batch::run_batch_dir(
        bundled_frontalface_cascade(),
        in_dir.path(),
        out_dir.path(),
        &BatchConfig::default(),
    )
    .expect("run_batch_dir");

    assert_eq!(stats.images_processed, 1);
    assert_eq!(
        stats.images_with_face, 1,
        "demo portrait must trigger ≥1 face"
    );
    assert!(stats.total_detections >= 1);
    assert_eq!(results.len(), 1);
    assert!(results[0].error.is_none());
    assert_eq!(results[0].width, 256);
    assert_eq!(results[0].height, 256);

    // Per-image annotated PNG written under detections/<input_stem>.png.
    let annotated = out_dir.path().join("detections").join("demo_face_256.png");
    assert!(
        annotated.is_file(),
        "annotated PNG missing at {}",
        annotated.display()
    );

    // Combined manifest written at the documented path.
    let manifest = batch::manifest_path(out_dir.path());
    assert!(manifest.is_file());
    let body = fs::read_to_string(&manifest).unwrap();
    assert!(
        body.contains("\"version\""),
        "manifest missing version key:\n{body}"
    );
    assert!(
        body.contains("\"images_processed\": 1"),
        "manifest wrong processed count:\n{body}"
    );
    assert!(
        body.contains("\"images_with_face\": 1"),
        "manifest wrong with-face count:\n{body}"
    );
    assert!(
        body.contains("demo_face_256"),
        "manifest missing the demo fixture name:\n{body}"
    );
}

#[test]
fn batch_dir_only_with_face_skips_empty_annotated_outputs_but_keeps_manifest_entry() {
    // Use a tiny blank PGM (all zeros) — guaranteed zero detections even with
    // a permissive cascade. Combined with the demo portrait, this exercises
    // both branches of `cfg.only_with_face` in one run.
    let in_dir = TempDir::new("rsface-batch-mixed-in").unwrap();
    let out_dir = TempDir::new("rsface-batch-mixed-out").unwrap();

    fs::copy(
        PathBuf::from(fixtures()).join("demo_face_256.pgm"),
        in_dir.path().join("a_face.pgm"),
    )
    .unwrap();

    // 16x16 black PGM (P5 header). Detector will score zero variance
    // everywhere → 0 detections regardless of cascade.
    let blank = b"P5\n16 16\n255\n".to_vec();
    let blank_body = vec![0u8; 16 * 16];
    let mut bytes = blank.clone();
    bytes.extend_from_slice(&blank_body);
    fs::write(in_dir.path().join("b_blank.pgm"), &bytes).unwrap();

    let cfg = BatchConfig {
        only_with_face: true,
        ..BatchConfig::default()
    };
    let (stats, results) = batch::run_batch_dir(
        bundled_frontalface_cascade(),
        in_dir.path(),
        out_dir.path(),
        &cfg,
    )
    .expect("run_batch_dir");

    assert_eq!(stats.images_processed, 2);
    assert_eq!(
        stats.images_with_face, 1,
        "only the demo portrait has faces"
    );

    // The annotated PNG for the blank image must NOT be on disk.
    let blank_anno = out_dir.path().join("detections").join("b_blank.png");
    assert!(
        !blank_anno.exists(),
        "only_with_face should skip empty annotated outputs"
    );

    // The demo portrait's annotated PNG SHOULD be on disk.
    let face_anno = out_dir.path().join("detections").join("a_face.png");
    assert!(face_anno.is_file());

    // Per-image results still describe both images (including the empty one).
    let blank_result = results
        .iter()
        .find(|r| r.input.ends_with("b_blank.pgm"))
        .unwrap();
    assert_eq!(blank_result.detections.len(), 0);
    assert!(
        blank_result.output.is_none(),
        "empty image has no output file"
    );
    assert!(blank_result.error.is_none());
}

#[test]
fn batch_dir_continues_after_a_per_image_decode_error() {
    let in_dir = TempDir::new("rsface-batch-err-in").unwrap();
    let out_dir = TempDir::new("rsface-batch-err-out").unwrap();

    fs::copy(
        PathBuf::from(fixtures()).join("demo_face_256.pgm"),
        in_dir.path().join("good.pgm"),
    )
    .unwrap();
    // Truncated PGM: header claims 256x256 but body is empty → decode fails
    // mid-stream. We don't promise an exact error message; we promise the
    // run completes and the bad entry has `error` set, the good one doesn't.
    fs::write(
        in_dir.path().join("bad.pgm"),
        b"P5\n256 256\n255\nthis-body-is-too-short",
    )
    .unwrap();

    let (stats, results) = batch::run_batch_dir(
        bundled_frontalface_cascade(),
        in_dir.path(),
        out_dir.path(),
        &BatchConfig::default(),
    )
    .expect("run_batch_dir");

    assert_eq!(stats.images_processed, 2);
    let bad = results
        .iter()
        .find(|r| r.input.ends_with("bad.pgm"))
        .unwrap();
    assert!(bad.error.is_some(), "truncated PGM must surface an error");
    assert!(bad.detections.is_empty());
    let good = results
        .iter()
        .find(|r| r.input.ends_with("good.pgm"))
        .unwrap();
    assert!(good.error.is_none(), "good fixture should pass cleanly");
    assert!(good.output.is_some());
}

#[test]
fn batch_dir_errors_when_input_directory_does_not_exist() {
    let out_dir = TempDir::new("rsface-batch-missing-out").unwrap();
    let missing =
        std::path::PathBuf::from("/this/path/should/never/exist/rsface-batch-missing-input");
    let err = batch::run_batch_dir(
        bundled_frontalface_cascade(),
        &missing,
        out_dir.path(),
        &BatchConfig::default(),
    )
    .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}
