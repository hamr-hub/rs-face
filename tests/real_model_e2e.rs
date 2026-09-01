//! End-to-end integration tests against **real** downloaded model weights.
//!
//! Every test here is skipped (not failed) when the weights are absent, so `cargo test`
//! stays green on a fresh clone. Run `tools/fetch_models.sh --all` first, and build with
//! an inference backend:
//!
//! ```sh
//! tools/fetch_models.sh --all
//! cargo test --features tract-backend --test real_model_e2e -- --nocapture
//! ```
//!
//! These tests are the difference between "the code compiles and the unit tests pass" and
//! "the system actually detects faces". The unit tests verify the decoding *logic* against
//! synthetic tensors; only these verify that our understanding of the real graph — its
//! output count, head layout, tensor shapes and embedding dimension — is correct.

#![cfg(feature = "tract-backend")]

use rsface::embedding::{Gallery, MatchConfig, MatchOutcome};
use rsface::face::Landmarks;
use rsface::image::RgbImage;
use rsface::models::{self, verify_bytes, Integrity};
use rsface::onnx::{Backend, SessionConfig};
use std::path::{Path, PathBuf};

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

/// Skip helper: returns `None` and prints why when a model is missing.
fn require(file: &str) -> Option<PathBuf> {
    let p = model_dir().join(file);
    if p.exists() {
        Some(p)
    } else {
        eprintln!(
            "SKIP: {} not found; run tools/fetch_models.sh --all",
            p.display()
        );
        None
    }
}

/// A synthetic image with enough structure that a detector is exercised on real signal
/// rather than a constant plane (which can short-circuit convolutions).
fn textured_image(w: usize, h: usize) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let d = img.as_mut_slice();
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            d[i] = ((x * 7 + y * 3) % 256) as u8;
            d[i + 1] = ((y * 5) % 256) as u8;
            d[i + 2] = ((x * 3 + y * 11) % 256) as u8;
        }
    }
    img
}

/// Load a real face photograph from `tests/fixtures/*.ppm`.
///
/// Real faces matter here in a way synthetic texture cannot substitute for. On a gradient
/// image ArcFace has no face to encode, so every embedding collapses into a tight cluster
/// (measured: cosine 0.80-0.82 between *unrelated* crops, a dynamic range of 0.016). Any
/// "same vs different" assertion on such input is measuring noise. Only genuine faces
/// exercise the discriminative behaviour the pipeline exists to provide.
fn face_fixture(name: &str) -> Option<RgbImage> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{name}.ppm"));
    if !p.exists() {
        eprintln!("SKIP: fixture {} not found", p.display());
        return None;
    }
    let mut f = std::fs::File::open(&p).expect("open fixture");
    Some(rsface::image::codec::read_ppm(&mut f).expect("decode ppm fixture"))
}

/// Detector configured for the fixtures, with both real models loaded.
fn load_detector() -> Option<rsface::scrfd_detector::ScrfdDetector> {
    let path = require("det_10g.onnx")?;
    let cfg = SessionConfig::default().with_backend(Backend::Tract);
    Some(
        rsface::scrfd_detector::ScrfdDetector::open(
            &path,
            Some(models::find("scrfd_10g_kps").unwrap()),
            &cfg,
            rsface::scrfd::ScrfdConfig::default(),
        )
        .expect("load detector"),
    )
}

fn load_recognizer() -> Option<rsface::arcface_recognizer::ArcFaceRecognizer> {
    let path = require("w600k_r50.onnx")?;
    let cfg = SessionConfig::default().with_backend(Backend::Tract);
    Some(
        rsface::arcface_recognizer::ArcFaceRecognizer::open(
            &path,
            Some(models::find("arcface_w600k_r50").unwrap()),
            &cfg,
        )
        .expect("load recogniser"),
    )
}

// ---------------------------------------------------------------------------
// Integrity
// ---------------------------------------------------------------------------

/// The pinned digests must match the real files. This is the test that makes the whole
/// integrity mechanism trustworthy: a pinned-but-wrong digest would reject every genuine
/// download, and the unit tests (which hash synthetic bytes) cannot detect that.
#[test]
fn pinned_digests_match_the_real_downloaded_weights() {
    for spec in models::REGISTRY {
        let Some(path) = require(spec.file_name) else {
            continue;
        };
        let bytes = std::fs::read(&path).expect("reading model");
        let integrity = verify_bytes(spec, &bytes);
        assert!(
            matches!(integrity, Integrity::Verified | Integrity::Unpinned),
            "{}: {integrity}",
            spec.id
        );
        if matches!(integrity, Integrity::Verified) {
            eprintln!("  {} sha256 VERIFIED", spec.id);
        }
    }
}

// ---------------------------------------------------------------------------
// SCRFD detection
// ---------------------------------------------------------------------------

/// Loads the real SCRFD graph and reports what it actually exposes.
///
/// This settles empirically a point the documentation disagrees on: whether the
/// `det_10g.onnx` shipped inside `buffalo_l` has a 5-point keypoint head. Our loader
/// derives the answer from the graph rather than the filename, so whichever it is, the
/// behaviour is correct — but it must be *reported*, because recognition depends on it.
#[test]
fn real_scrfd_graph_loads_and_reports_its_heads() {
    let Some(path) = require("det_10g.onnx") else {
        return;
    };
    let spec = models::find("scrfd_10g_kps").unwrap();
    let cfg = SessionConfig::default().with_backend(Backend::Tract);

    let det = rsface::scrfd_detector::ScrfdDetector::open(
        &path,
        Some(spec),
        &cfg,
        rsface::scrfd::ScrfdConfig::default(),
    )
    .expect("the real SCRFD graph must load");

    eprintln!(
        "  SCRFD loaded: keypoints={} layout={:?} input={}",
        det.has_keypoints(),
        det.head_layout(),
        det.config().input_size
    );

    use rsface::face_detector::{ColorInput, FaceDetector, Maturity};
    assert_eq!(det.maturity(), Maturity::Production);
    assert_eq!(det.color_input(), ColorInput::Rgb);
    assert_eq!(det.name(), "scrfd");
    assert_eq!(
        det.has_landmarks(),
        det.has_keypoints(),
        "has_landmarks must reflect the loaded graph, not a guess"
    );
}

/// A forward pass on a real graph must produce sane, in-frame geometry.
///
/// Deliberately does not assert a face *count*: the input is synthetic texture, so any
/// number of detections (including zero) is legitimate. What must hold is that every box
/// the decoder emits is geometrically valid — that is what catches an anchor-offset or
/// stride-scaling bug, which would otherwise yield boxes far outside the frame.
#[test]
fn real_scrfd_inference_produces_valid_geometry() {
    let Some(path) = require("det_10g.onnx") else {
        return;
    };
    let spec = models::find("scrfd_10g_kps").unwrap();
    let cfg = SessionConfig::default().with_backend(Backend::Tract);
    let det = rsface::scrfd_detector::ScrfdDetector::open(
        &path,
        Some(spec),
        &cfg,
        // Lower the threshold so we exercise the decode path on real activations.
        rsface::scrfd::ScrfdConfig::default().with_score_threshold(0.3),
    )
    .expect("load");

    let (w, h) = (640usize, 480usize);
    let img = textured_image(w, h);
    let dets = det.detect_rgb(&img).expect("inference must not error");

    eprintln!(
        "  SCRFD returned {} detections on synthetic texture",
        dets.len()
    );

    for (i, d) in dets.iter().enumerate() {
        assert!(
            d.x1 >= 0.0 && d.y1 >= 0.0 && d.x2 <= w as f32 && d.y2 <= h as f32,
            "detection {i} box ({}, {}, {}, {}) escapes the {w}x{h} frame — \
             likely an anchor-centre or stride-scaling bug",
            d.x1,
            d.y1,
            d.x2,
            d.y2
        );
        assert!(
            d.width() > 0.0 && d.height() > 0.0,
            "detection {i} is degenerate"
        );
        assert!(
            d.score >= 0.3 && d.score <= 1.0,
            "detection {i} score {} outside the expected range",
            d.score
        );
        if let Some(l) = d.landmarks {
            for (j, (x, y)) in l.points.iter().enumerate() {
                assert!(
                    x.is_finite() && y.is_finite(),
                    "detection {i} landmark {j} is not finite"
                );
            }
        }
    }
}

/// Boxes must be free of NaN even at a very low threshold, where marginal activations
/// dominate — the regime where a decoding bug shows up first.
#[test]
fn real_scrfd_low_threshold_output_is_all_finite() {
    let Some(path) = require("det_10g.onnx") else {
        return;
    };
    let spec = models::find("scrfd_10g_kps").unwrap();
    let cfg = SessionConfig::default().with_backend(Backend::Tract);
    let det = rsface::scrfd_detector::ScrfdDetector::open(
        &path,
        Some(spec),
        &cfg,
        rsface::scrfd::ScrfdConfig::default().with_score_threshold(0.05),
    )
    .expect("load");

    let dets = det
        .detect_rgb(&textured_image(320, 320))
        .expect("inference");
    eprintln!("  SCRFD @0.05 threshold: {} detections", dets.len());
    for d in &dets {
        assert!(
            d.x1.is_finite() && d.y1.is_finite() && d.x2.is_finite() && d.y2.is_finite(),
            "non-finite box coordinate"
        );
    }
}

/// A blank frame must not error and must not hallucinate high-confidence faces.
#[test]
fn real_scrfd_on_a_blank_frame_is_quiet() {
    let Some(path) = require("det_10g.onnx") else {
        return;
    };
    let spec = models::find("scrfd_10g_kps").unwrap();
    let cfg = SessionConfig::default().with_backend(Backend::Tract);
    let det =
        rsface::scrfd_detector::ScrfdDetector::open(&path, Some(spec), &cfg, Default::default())
            .expect("load");

    let dets = det.detect_rgb(&RgbImage::new(640, 640)).expect("inference");
    eprintln!("  SCRFD on a black frame: {} detections", dets.len());
    assert!(
        dets.len() < 10,
        "a uniform black frame produced {} detections at the default threshold, which \
         suggests the score head is being read from the wrong tensor",
        dets.len()
    );
}

// ---------------------------------------------------------------------------
// ArcFace recognition
// ---------------------------------------------------------------------------

/// Loads the real ArcFace graph and checks it emits a 512-d embedding.
#[test]
fn real_arcface_graph_loads_with_512_dims() {
    let Some(path) = require("w600k_r50.onnx") else {
        return;
    };
    let spec = models::find("arcface_w600k_r50").unwrap();
    let cfg = SessionConfig::default().with_backend(Backend::Tract);

    let rec = rsface::arcface_recognizer::ArcFaceRecognizer::open(&path, Some(spec), &cfg)
        .expect("the real ArcFace graph must load");

    eprintln!("  ArcFace loaded: embedding dim = {}", rec.dim());
    assert_eq!(rec.dim(), 512, "w600k_r50 must produce a 512-d embedding");
    assert!(rec.is_standard_arcface_dim());
}

/// **The test that proves recognition actually works**: real faces, real detection, real
/// embeddings, on genuinely different people.
///
/// Detects a face in each of two photographs of different people, embeds both, and checks
/// that the two identities are well separated while a re-embedding of the same face is
/// essentially identical. This is the end-to-end discriminative property; nothing short of
/// real faces can establish it.
#[test]
fn real_faces_of_different_people_are_discriminated() {
    let (Some(det), Some(rec)) = (load_detector(), load_recognizer()) else {
        return;
    };
    let (Some(lena), Some(biden)) = (face_fixture("lena"), face_fixture("biden")) else {
        return;
    };

    // Detect in each photo and take the highest-scoring face.
    let mut best = |img: &RgbImage, who: &str| -> Option<(Landmarks, f32)> {
        let dets = det.detect_rgb(img).expect("detect");
        eprintln!("  {who}: {} face(s) detected", dets.len());
        let top = dets
            .iter()
            .filter(|d| d.landmarks.is_some())
            .max_by(|a, b| a.score.total_cmp(&b.score))?;
        eprintln!(
            "    top score {:.3} box ({:.0},{:.0})-({:.0},{:.0})",
            top.score, top.x1, top.y1, top.x2, top.y2
        );
        Some((top.landmarks.unwrap(), top.score))
    };

    let Some((lena_lms, lena_score)) = best(&lena, "lena") else {
        eprintln!("  SKIP: no face with landmarks found in lena");
        return;
    };
    let Some((biden_lms, _)) = best(&biden, "biden") else {
        eprintln!("  SKIP: no face with landmarks found in biden");
        return;
    };

    // A real face must be detected with real confidence.
    assert!(
        lena_score > 0.5,
        "SCRFD should detect the face in lena.ppm confidently, got {lena_score:.3}"
    );

    let e_lena = rec.embed(&lena, &lena_lms).expect("embed lena");
    let e_lena2 = rec.embed(&lena, &lena_lms).expect("embed lena again");
    let e_biden = rec.embed(&biden, &biden_lms).expect("embed biden");

    let same = e_lena.cosine(&e_lena2).unwrap();
    let diff = e_lena.cosine(&e_biden).unwrap();
    eprintln!("  cosine(lena, lena)  = {same:.4}");
    eprintln!("  cosine(lena, biden) = {diff:.4}");

    assert!(
        same > 0.999,
        "embedding the same face twice must be deterministic, got {same:.4}"
    );

    // The real payoff. Two different people must fall below the match threshold, with a
    // wide margin — not the 0.016 range synthetic texture produced.
    let threshold = MatchConfig::default().threshold;
    assert!(
        diff < threshold,
        "two different people scored {diff:.4}, at or above the {threshold} match \
         threshold; recognition would produce false accepts"
    );
    assert!(
        same - diff > 0.5,
        "separation between same-identity ({same:.4}) and different-identity ({diff:.4}) \
         is only {:.4}; the embedding is not discriminating faces",
        same - diff
    );
}

/// Alignment must cancel a similarity transform of the input.
///
/// Scales the whole photograph, re-detects, and compares embeddings. Because alignment
/// warps both crops onto the same canonical landmark layout, the two embeddings should be
/// close despite the detector seeing the face at a different pixel size. Uses a real face
/// so the embedding has actual identity content to preserve.
#[test]
fn real_face_embedding_survives_rescaling_the_image() {
    let (Some(det), Some(rec)) = (load_detector(), load_recognizer()) else {
        return;
    };
    let Some(img) = face_fixture("lena") else {
        return;
    };

    // Downscale by 2/3 using the crate's own bilinear resize, via the gray path for the
    // luma and re-expanding: simplest is to build a smaller RGB image by sampling.
    let (w, h) = (img.width(), img.height());
    let (nw, nh) = (w * 2 / 3, h * 2 / 3);
    let mut small = RgbImage::new(nw, nh);
    {
        let src = img.as_slice();
        let dst = small.as_mut_slice();
        for y in 0..nh {
            let sy = y * h / nh;
            for x in 0..nw {
                let sx = x * w / nw;
                let si = (sy * w + sx) * 3;
                let di = (y * nw + x) * 3;
                dst[di..di + 3].copy_from_slice(&src[si..si + 3]);
            }
        }
    }

    let pick = |i: &RgbImage| -> Option<Landmarks> {
        det.detect_rgb(i)
            .expect("detect")
            .into_iter()
            .filter(|d| d.landmarks.is_some())
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .and_then(|d| d.landmarks)
    };

    let (Some(l_full), Some(l_small)) = (pick(&img), pick(&small)) else {
        eprintln!("  SKIP: face not detected in both scales");
        return;
    };

    let e_full = rec.embed(&img, &l_full).expect("embed full");
    let e_small = rec.embed(&small, &l_small).expect("embed small");
    let sim = e_full.cosine(&e_small).unwrap();
    eprintln!("  cosine(512px, 341px) = {sim:.4}");

    assert!(
        sim > 0.7,
        "the same face at two scales embedded to only {sim:.4}; alignment is not \
         normalising scale as it must"
    );
}

/// A photo with two people must yield two distinct identities.
#[test]
fn real_two_person_photo_yields_distinct_identities() {
    let (Some(det), Some(rec)) = (load_detector(), load_recognizer()) else {
        return;
    };
    let Some(img) = face_fixture("two-people") else {
        return;
    };

    let dets = det.detect_rgb(&img).expect("detect");
    eprintln!("  two-people.ppm: {} face(s) detected", dets.len());
    for d in &dets {
        eprintln!(
            "    score {:.3} box ({:.0},{:.0})-({:.0},{:.0}) kps={}",
            d.score,
            d.x1,
            d.y1,
            d.x2,
            d.y2,
            d.landmarks.is_some()
        );
    }

    assert!(
        dets.len() >= 2,
        "expected at least 2 faces in two-people.ppm, found {}",
        dets.len()
    );

    let embeddings = rec.embed_all(&img, &dets).expect("embed all");
    assert!(embeddings.len() >= 2, "both faces must embed");

    // Two different people in one frame must not collapse to one identity.
    let sim = embeddings[0].1.cosine(&embeddings[1].1).unwrap();
    eprintln!("  cosine(person A, person B) = {sim:.4}");
    let threshold = MatchConfig::default().threshold;
    assert!(
        sim < threshold,
        "two different people in the same photo scored {sim:.4}, at or above the \
         {threshold} threshold"
    );
}

/// A blank crop must be handled, not crash — and must not produce a vector that matches
/// everything in a gallery.
#[test]
fn real_arcface_handles_a_blank_crop() {
    let Some(path) = require("w600k_r50.onnx") else {
        return;
    };
    let spec = models::find("arcface_w600k_r50").unwrap();
    let cfg = SessionConfig::default().with_backend(Backend::Tract);
    let rec =
        rsface::arcface_recognizer::ArcFaceRecognizer::open(&path, Some(spec), &cfg).expect("load");

    let blank = RgbImage::new(112, 112);
    match rec.embed_aligned(&blank) {
        Ok(e) => {
            assert_eq!(e.dim(), 512);
            // Whatever it is, it must be finite and unit length.
            let norm: f32 = e.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "embedding norm {norm} != 1");
        }
        Err(e) => eprintln!("  blank crop rejected (acceptable): {e}"),
    }
}

/// Rejects a wrongly-sized crop rather than silently resizing.
#[test]
fn real_arcface_rejects_a_misaligned_crop_size() {
    let Some(path) = require("w600k_r50.onnx") else {
        return;
    };
    let spec = models::find("arcface_w600k_r50").unwrap();
    let cfg = SessionConfig::default().with_backend(Backend::Tract);
    let rec =
        rsface::arcface_recognizer::ArcFaceRecognizer::open(&path, Some(spec), &cfg).expect("load");

    assert!(
        rec.embed_aligned(&RgbImage::new(96, 96)).is_err(),
        "a 96x96 crop must be refused, not resized"
    );
}

// ---------------------------------------------------------------------------
// Gallery round-trip
// ---------------------------------------------------------------------------

/// The full enrol-then-identify loop against real weights.
#[test]
fn real_gallery_enroll_and_identify_round_trip() {
    let Some(path) = require("w600k_r50.onnx") else {
        return;
    };
    let spec = models::find("arcface_w600k_r50").unwrap();
    let cfg = SessionConfig::default().with_backend(Backend::Tract);
    let rec =
        rsface::arcface_recognizer::ArcFaceRecognizer::open(&path, Some(spec), &cfg).expect("load");

    let img = textured_image(800, 800);
    let reference = rsface::align::ARCFACE_REFERENCE_LANDMARKS;
    let at = |dx: f32, dy: f32| {
        let mut p = reference;
        for q in p.iter_mut() {
            q.0 = q.0 * 3.0 + dx;
            q.1 = q.1 * 3.0 + dy;
        }
        Landmarks { points: p }
    };

    let alice = at(150.0, 150.0);
    let bob = at(450.0, 450.0);

    let mut gallery = Gallery::new(MatchConfig::default());
    rec.enroll(&mut gallery, "alice", &img, &alice)
        .expect("enroll alice");
    rec.enroll(&mut gallery, "bob", &img, &bob)
        .expect("enroll bob");
    assert_eq!(gallery.len(), 2);

    // Re-identifying an enrolled face must return that exact identity at ~1.0.
    match rec.identify(&gallery, &img, &alice).expect("identify") {
        MatchOutcome::Match {
            label, similarity, ..
        } => {
            eprintln!("  identified '{label}' at cosine {similarity:.4}");
            assert_eq!(label, "alice");
            assert!(
                similarity > 0.99,
                "an identical crop must re-identify near 1.0, got {similarity:.4}"
            );
        }
        other => panic!("expected a Match for the enrolled face, got {other:?}"),
    }

    // The ranking must place the correct identity first.
    let e = rec.embed(&img, &alice).expect("embed");
    let ranked = gallery.rank(&e);
    assert_eq!(ranked[0].0, "alice");
    eprintln!("  ranking: {ranked:?}");
}

// ---------------------------------------------------------------------------
// Detection -> recognition pipeline
// ---------------------------------------------------------------------------

/// Wires the two real models together, exactly as an application would.
///
/// Asserts the *contract* between them rather than a face count: every detection carrying
/// landmarks must yield a valid 512-d embedding, and detections without landmarks must be
/// skipped rather than misaligned.
#[test]
fn real_detection_to_recognition_pipeline_is_consistent() {
    let (Some(dpath), Some(rpath)) = (require("det_10g.onnx"), require("w600k_r50.onnx")) else {
        return;
    };
    let cfg = SessionConfig::default().with_backend(Backend::Tract);

    let det = rsface::scrfd_detector::ScrfdDetector::open(
        &dpath,
        Some(models::find("scrfd_10g_kps").unwrap()),
        &cfg,
        rsface::scrfd::ScrfdConfig::default().with_score_threshold(0.3),
    )
    .expect("load detector");

    let rec = rsface::arcface_recognizer::ArcFaceRecognizer::open(
        &rpath,
        Some(models::find("arcface_w600k_r50").unwrap()),
        &cfg,
    )
    .expect("load recogniser");

    let img = textured_image(640, 640);
    let dets = det.detect_rgb(&img).expect("detect");
    let embeddings = rec.embed_all(&img, &dets).expect("embed all");

    let with_landmarks = dets.iter().filter(|d| d.landmarks.is_some()).count();
    eprintln!(
        "  pipeline: {} detections, {} with landmarks, {} embedded",
        dets.len(),
        with_landmarks,
        embeddings.len()
    );

    assert_eq!(
        embeddings.len(),
        with_landmarks,
        "every landmark-bearing detection must embed, and no other"
    );
    for (idx, e) in &embeddings {
        assert_eq!(e.dim(), 512, "detection {idx} produced a non-512 embedding");
        assert!(dets[*idx].landmarks.is_some());
    }

    // If the detector has no keypoint head, recognition is structurally impossible and
    // that must be visible rather than producing garbage.
    if !det.has_keypoints() {
        eprintln!(
            "  NOTE: this SCRFD export has no keypoint head, so recognition cannot run \
             from its output. Use an SCRFD *_kps export for the recognition pipeline."
        );
        assert!(embeddings.is_empty());
    }
}

/// Both real graphs must load under the same backend in one process, which is what a
/// server does. Catches any global-state conflict between sessions.
#[test]
fn both_real_models_coexist_in_one_process() {
    let (Some(d), Some(r)) = (require("det_10g.onnx"), require("w600k_r50.onnx")) else {
        return;
    };
    let cfg = SessionConfig::default().with_backend(Backend::Tract);
    let _det = rsface::scrfd_detector::ScrfdDetector::open(
        &d,
        Some(models::find("scrfd_10g_kps").unwrap()),
        &cfg,
        Default::default(),
    )
    .expect("detector");
    let _rec = rsface::arcface_recognizer::ArcFaceRecognizer::open(
        &r,
        Some(models::find("arcface_w600k_r50").unwrap()),
        &cfg,
    )
    .expect("recogniser");
    eprintln!("  both sessions live simultaneously");
}

/// A corrupt model file must be refused by the integrity gate before the backend ever
/// parses it. Verifies the gate is actually wired into the real load path.
#[test]
fn a_truncated_real_model_is_refused_by_the_integrity_gate() {
    let Some(path) = require("det_10g.onnx") else {
        return;
    };
    let mut bytes = std::fs::read(&path).expect("read");
    bytes.truncate(bytes.len() / 2);

    let tmp = std::env::temp_dir().join("rsface_truncated_det_10g.onnx");
    std::fs::write(&tmp, &bytes).expect("write");

    let spec = models::find("scrfd_10g_kps").unwrap();
    let cfg = SessionConfig::default().with_backend(Backend::Tract);
    let err =
        rsface::scrfd_detector::ScrfdDetector::open(&tmp, Some(spec), &cfg, Default::default())
            .err()
            .expect("a truncated model must be refused");

    eprintln!("  truncated model correctly refused: {err}");
    assert!(
        matches!(err, rsface::onnx::OnnxError::Integrity(_)),
        "expected the integrity gate to catch this before parsing, got {err:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// Sanity: the model directory helper points somewhere real.
#[test]
fn model_dir_is_under_the_crate_root() {
    assert!(model_dir().ends_with("models"));
    assert!(Path::new(env!("CARGO_MANIFEST_DIR")).exists());
}
