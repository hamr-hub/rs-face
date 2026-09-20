//! Dispatch the same probe through every zero-dep recogniser — the
//! recognition-side counterpart of `detect_uniform`.
//!
//! LBPH, eigenfaces and Fisherfaces all implement the `FaceRecognizer`
//! trait, so benchmark / platform code can own a heterogeneous
//! `Vec<Box<dyn FaceRecognizer>>` and query each one with the same call
//! shape. Here we train all three on a tiny synthetic gallery (2 shots
//! per identity) and identify an exact gallery member through the trait.
//!
//! Run with:
//!   cargo run --release --example recognise_uniform

use rsface::eigenface::{EigenfaceConfig, EigenfaceRecognizer};
use rsface::fisherface::{FisherfaceConfig, FisherfaceRecognizer};
use rsface::image::GrayImage;
use rsface::lbph::{LbphConfig, LbphRecognizer};
use rsface::recognizer::FaceRecognizer;

fn synthetic_crop(seed: u32) -> GrayImage {
    let mut img = GrayImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            let dx = (x as f32 - 32.0).abs();
            let dy = (y as f32 - 32.0).abs();
            let ring = (dx * dx + dy * dy).sqrt();
            let base = if ring < 18.0 { 220.0 } else { 20.0 };
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
    // Gallery: three identities, two shots each. Keep the originals around
    // because the subspace recognisers borrow their training samples.
    let crops = [
        ("alice", synthetic_crop(1)),
        ("alice", synthetic_crop(2)),
        ("bob", synthetic_crop(9)),
        ("bob", synthetic_crop(10)),
        ("carol", synthetic_crop(17)),
        ("carol", synthetic_crop(18)),
    ];
    let samples: Vec<(&str, &GrayImage)> = crops.iter().map(|(l, i)| (*l, i)).collect();
    let probe = synthetic_crop(1); // exact replica of alice's first shot

    // LBPH is incremental: descriptors are appended one by one.
    let mut lbph = LbphRecognizer::new(LbphConfig::default());
    for (label, crop) in &samples {
        lbph.enroll(*label, crop);
    }

    // The subspace recognisers are train-once values (adding a person
    // changes the PCA/LDA basis — retrain instead of mutating).
    let eigen = EigenfaceRecognizer::train(EigenfaceConfig::default(), samples.clone())
        .expect("eigenfaces train");
    let fisher = FisherfaceRecognizer::train(FisherfaceConfig::default(), samples)
        .expect("fisherfaces train");

    // Uniform dispatch: one heterogeneous collection, one call shape.
    let recognisers: Vec<Box<dyn FaceRecognizer>> =
        vec![Box::new(lbph), Box::new(eigen), Box::new(fisher)];

    for rec in &recognisers {
        let outcome = rec.identify_crop(&probe);
        match outcome.label() {
            Some(label) => println!(
                "{:<10}  match={:<6}  distance={:.3}  ({} identities, {} crops)",
                rec.name(),
                label,
                outcome.distance().unwrap_or(f32::NAN),
                rec.len(),
                rec.crop_count(),
            ),
            None => println!("{:<10}  no match: {:?}", rec.name(), outcome),
        }
    }
}
