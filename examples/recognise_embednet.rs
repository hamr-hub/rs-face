//! Zero-dependency deep metric learning: train EmbedNet on synthetic
//! identities, enrol them, and identify a probe — no ONNX, no C++
//! runtime, no third-party crates.
//!
//! This is the pure-Rust counterpart to the ONNX ArcFace path
//! (`examples/detect_scrfd_arcface.rs`). Synthetic sinusoid "identities"
//! stand in for real crops only because the crate cannot ship face data;
//! the mechanics — im2col convs, full backprop verified against finite
//! differences, contrastive-pair Adam — are the same ones the
//! `embednet_train` binary runs on `crops/<person>/*.pgm` directories.
//!
//! Run with:
//!   cargo run --release --example recognise_embednet

use rsface::embedding::Embedding;
use rsface::embednet::toy::identity;
use rsface::embednet::{ContrastiveTrainer, EmbedNet, EmbedNetConfig, EmbedNetRecognizer, Rng};
use rsface::image::GrayImage;
use rsface::recognizer::{FaceRecognizer, IncrementalRecognizer, Recognition};

const IDENTITIES: u64 = 4;
const STEPS: usize = 400;

fn patch_to_crop(patch: &[f32]) -> GrayImage {
    let mut g = GrayImage::new(56, 56);
    for (i, v) in patch.iter().enumerate() {
        g.as_mut_slice()[i] = (v * 255.0).clamp(0.0, 255.0) as u8;
    }
    g
}

fn distance(net: &EmbedNet, a: &[f32], b: &[f32]) -> f32 {
    let ea = Embedding::from_raw(&net.forward_raw(a)).unwrap();
    let eb = Embedding::from_raw(&net.forward_raw(b)).unwrap();
    ea.sq_euclidean(&eb).unwrap().sqrt()
}

fn main() {
    // 1. Four synthetic "people", each a stable intensity field whose
    //    samples get translation/brightness/contrast/noise jitter.
    let ids: Vec<_> = (0..IDENTITIES).map(identity).collect();
    let mut trainer = ContrastiveTrainer::new(2026);
    let mut rng = Rng::new(777);

    // 2. Contrastive training: alternating same/different pairs.
    println!("training EmbedNet on {IDENTITIES} synthetic identities ({STEPS} steps)...");
    for step in 0..STEPS {
        let i = (step as u64) % IDENTITIES;
        let same = step % 2 == 0;
        let a = ids[i as usize].sample(&mut rng);
        let b = if same {
            ids[i as usize].sample(&mut rng)
        } else {
            let j = ((i + 1) % IDENTITIES) as usize;
            ids[j].sample(&mut rng)
        };
        let loss = trainer.train_step(&a, &b, same);
        if step % 100 == 0 {
            println!("  step {step:>4}  loss={loss:.4}");
        }
    }
    let net = trainer.into_net();

    // 3. Held-out separation check on freshly jittered samples.
    let mut eval = Rng::new(0xABCD);
    let held: Vec<Vec<f32>> = ids.iter().map(|id| id.sample(&mut eval)).collect();
    let d_same = distance(&net, &held[0], &ids[0].sample(&mut Rng::new(4242)));
    let d_diff = distance(&net, &held[0], &held[1]);
    println!("held-out D(same)={d_same:.3}  D(different)={d_diff:.3}");
    assert!(
        d_same < d_diff,
        "metric learning failed: same pair not closer than a different pair"
    );

    // 4. Enrol every identity into the uniform FaceRecognizer adapter
    //    (keep a clone so the trained weights can be persisted below).
    let config = EmbedNetConfig {
        distance_threshold: 0.9,
        min_distance_margin: 0.0,
    };
    let mut rec = EmbedNetRecognizer::with_config(net.clone(), config);
    let mut enroll_rng = Rng::new(9090);
    for (i, id) in ids.iter().enumerate() {
        for _ in 0..3 {
            rec.enroll(
                format!("person{i}"),
                &patch_to_crop(&id.sample(&mut enroll_rng)),
            );
        }
    }
    println!(
        "gallery: {} identities, {} embedding(s)",
        rec.len(),
        rec.crop_count()
    );

    // 5. Identify a fresh probe of person 0.
    let probe_patch = ids[0].sample(&mut Rng::new(31337));
    let probe = patch_to_crop(&probe_patch);
    match rec.identify_crop(&probe) {
        Recognition::Match {
            label,
            distance,
            margin,
        } => println!("identify -> {label}  D={distance:.3}  margin={margin:.3}"),
        other => panic!("expected Match for held-out same-identity probe, got {other:?}"),
    }

    // 6. Persistence: save the trained `.rsen` weights, reload, and prove
    //    the forward output is bit-identical.
    let path = std::env::temp_dir().join(format!(
        "rsface_embednet_example_{}.rsen",
        std::process::id()
    ));
    net.save(&path).expect("save .rsen");
    let reloaded = EmbedNet::load(&path).expect("load .rsen");
    assert_eq!(
        net.forward_raw(&EmbedNet::preprocess(&probe)),
        reloaded.forward_raw(&EmbedNet::preprocess(&probe))
    );
    std::fs::remove_file(&path).expect("cleanup .rsen");
    println!("trained weights round-trip through .rsen OK");
    println!("example OK");
}
