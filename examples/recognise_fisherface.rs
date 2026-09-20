//! Train a fisherfaces (LDA) recognizer on a tiny gallery and identify a probe.
//!
//! Fisherfaces is the discriminative zero-dep recognizer — after an n−C PCA
//! reduction it learns the at-most-C−1 directions that separate the enrolled
//! identities while keeping each identity's crops together (two Jacobi
//! eigendecompositions in pure `std`). Like eigenfaces the model is
//! gallery-specific and must be retrained when identities change.
//!
//! This example trains on synthetic crops so it runs out of the box; for the
//! measured accuracy numbers see `docs/recognition-fisherface.md`.
//!
//! Run with:
//!   cargo run --release --example recognise_fisherface

use rsface::fisherface::{FisherMatch, FisherfaceConfig, FisherfaceRecognizer};
use rsface::image::GrayImage;

fn synthetic_crop(seed: u32) -> GrayImage {
    let mut img = GrayImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            let dx = (x as f32 - 32.0).abs();
            let dy = (y as f32 - 32.0).abs();
            let ring = (dx * dx + dy * dy).sqrt();
            let base = if ring < 18.0 { 220.0 } else { 20.0 };
            // Each identity's "signature" is a deterministic offset in the
            // dark region, so the two synthetic identities are separable.
            let mut v = base + (seed as f32) * 4.0;
            if ring > 24.0 && ring < 28.0 {
                v = 120.0 + (seed as f32) * 20.0;
            }
            img[(x, y)] = v.clamp(0.0, 255.0) as u8;
        }
    }
    img
}

fn main() {
    // 1. Build a tiny labelled gallery (2 identities × 2 crops each).
    //    At least one identity must appear more than once, or train returns
    //    FisherfaceError::DegenerateGallery (n − C = 0 leaves no within-class
    //    subspace). Samples are borrowed; the crops must outlive the recogniser.
    let a1 = synthetic_crop(1);
    let a2 = synthetic_crop(2);
    let b1 = synthetic_crop(9);
    let b2 = synthetic_crop(10);
    let samples: Vec<(&str, &GrayImage)> =
        vec![("alice", &a1), ("alice", &a2), ("bob", &b1), ("bob", &b2)];

    // 2. Train: PCA reduction to n−C directions, then at most C−1 Fisher axes.
    let rec = match FisherfaceRecognizer::train(FisherfaceConfig::default(), samples) {
        Ok(rec) => rec,
        Err(e) => {
            eprintln!("fisherfaces train failed: {e:?}");
            std::process::exit(2);
        }
    };
    println!(
        "fisherfaces: {} crops, {} identities, {} discriminant component(s)",
        rec.crop_count(),
        rec.len(),
        rec.component_count(),
    );

    // 3. Identify a probe.
    let probe = synthetic_crop(1);
    match rec.identify_crop(&probe) {
        FisherMatch::Match {
            label,
            distance,
            margin,
        } => {
            println!("fisherfaces  ->  match={label}  distance={distance:.3}  margin={margin:.3}");
        }
        FisherMatch::BelowThreshold { best } => {
            println!("fisherfaces  ->  no match within threshold (best={best:?})");
        }
        FisherMatch::Ambiguous {
            first,
            second,
            margin,
        } => {
            println!(
                "fisherfaces  ->  ambiguous between {first} and {second} (margin={margin:.3})"
            );
        }
        FisherMatch::NoCandidates => {
            println!("fisherfaces  ->  no identities enrolled");
        }
    }

    // 4. Threshold-free close-set ranking — the recommended primary API:
    if let Some((label, distance)) = rec.rank_crop(&probe).first() {
        println!("fisherfaces rank -> {label} (d={distance:.3})");
    }

    // 5. Verification: does this crop look like alice?
    let score = rec.verify("alice", &probe).unwrap_or(f32::INFINITY);
    println!("fisherfaces verify(alice, probe) = {score:.3}");

    // 6. Persistence: the trained model (mean, Fisher axes, projections)
    //    survives a restart as the zero-dependency `RSLD` blob — no retrain
    //    and no original crops needed to reload. Reload is bit-exact.
    let model_path = std::env::temp_dir().join(format!(
        "rsface_example_fisher_{}_{}.fisher",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    rec.save(&model_path).expect("save fisherfaces model");
    let before = rec.rank_crop(&probe);
    drop(rec);

    let reloaded = FisherfaceRecognizer::load(&model_path).expect("load fisherfaces model");
    println!(
        "fisherfaces model reloaded: {} identities, {} crop(s)",
        reloaded.len(),
        reloaded.crop_count()
    );
    assert_eq!(reloaded.rank_crop(&probe), before);
    std::fs::remove_file(&model_path).expect("cleanup model");
    println!("fisherfaces persistence round-trip OK (ranking unchanged)");
}
