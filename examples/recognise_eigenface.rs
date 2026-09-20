//! Train an eigenfaces (PCA) recognizer on a tiny gallery and identify a probe.
//!
//! Eigenfaces is the other zero-dep recognizer — learn the gallery's variance
//! via a Gram-matrix eigendecomposition (Jacobi, in pure `std`) and match by
//! projection. The model is *gallery-specific* and must be retrained when
//! identities are added.
//!
//! This example trains on synthetic crops so it runs out of the box; for a
//! real accuracy demo see `docs/recognition-eigenface.md`.
//!
//! Run with:
//!   cargo run --release --example recognise_eigenface

use rsface::eigenface::{EigenMatch, EigenfaceConfig, EigenfaceRecognizer};
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
    //    EigenfaceRecognizer::train borrows the samples by reference so the
    //    gallery crops must outlive the recogniser.
    let a1 = synthetic_crop(1);
    let a2 = synthetic_crop(2);
    let b1 = synthetic_crop(9);
    let b2 = synthetic_crop(10);
    let samples: Vec<(&str, &GrayImage)> =
        vec![("alice", &a1), ("alice", &a2), ("bob", &b1), ("bob", &b2)];

    // 2. Train. Eigenfaces needs at least 2 samples and at least 2 distinct
    //    labels — EigenfaceError surfaces the failure modes so we can map
    //    them into friendly messages.
    let rec = match EigenfaceRecognizer::train(EigenfaceConfig::default(), samples) {
        Ok(rec) => rec,
        Err(e) => {
            eprintln!("eigenfaces train failed: {e:?}");
            std::process::exit(2);
        }
    };

    // 3. Identify a probe.
    let probe = synthetic_crop(1);
    match rec.identify_crop(&probe) {
        EigenMatch::Match {
            label,
            distance,
            margin,
        } => {
            println!("eigenfaces  ->  match={label}  distance={distance:.3}  margin={margin:.3}");
        }
        EigenMatch::BelowThreshold { best } => {
            println!("eigenfaces  ->  no match within threshold (best={best:?})");
        }
        EigenMatch::Ambiguous {
            first,
            second,
            margin,
        } => {
            println!("eigenfaces  ->  ambiguous between {first} and {second} (margin={margin:.3})");
        }
        EigenMatch::NoCandidates => {
            println!("eigenfaces  ->  no identities enrolled");
        }
    }

    // 4. Verification: does this crop look like alice?
    let score = rec.verify("alice", &probe).unwrap_or(f32::INFINITY);
    println!("eigenfaces verify(alice, probe) = {score:.3}");

    // 5. Persistence: the trained model (mean, eigenvectors, projections)
    //    survives a restart as the zero-dependency `RSEF` blob — no retrain
    //    and no original crops needed to reload. Reload is bit-exact.
    let model_path = std::env::temp_dir().join(format!(
        "rsface_example_eigen_{}_{}.eigen",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    rec.save(&model_path).expect("save eigenfaces model");
    let before = rec.rank_crop(&probe);
    drop(rec);

    let reloaded = EigenfaceRecognizer::load(&model_path).expect("load eigenfaces model");
    println!(
        "eigenfaces model reloaded: {} identities, {} crop(s)",
        reloaded.len(),
        reloaded.crop_count()
    );
    assert_eq!(reloaded.rank_crop(&probe), before);
    std::fs::remove_file(&model_path).expect("cleanup model");
    println!("eigenfaces persistence round-trip OK (ranking unchanged)");
}
