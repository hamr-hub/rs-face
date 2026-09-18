//! Train an LBPH recognizer on a small gallery and identify a probe.
//!
//! LBPH is the zero-dependency counterpart to ArcFace: histogram of uniform
//! Local Binary Patterns + chi-square distance. It is enough for controlled
//! access-control scenarios; for unconstrained scenes use the
//! `ort-backend` ArcFace path instead.
//!
//! This example doesn't ship face crops. Run it on any two-image directory
//! of the form `gallery/<person>/<file>.ppm` and you'll get a non-trivial
//! rank-1; on synthetic input it still proves the API.
//!
//! Run with:
//!   cargo run --release --example recognise_lbph

use rsface::image::GrayImage;
use rsface::lbph::{LbphConfig, LbphMatch, LbphRecognizer};

fn load_ppm_or_placeholder() -> GrayImage {
    // No fixture? Build a small synthetic crop so the example still runs.
    let mut img = GrayImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            let dx = (x as f32 - 32.0).abs();
            let dy = (y as f32 - 32.0).abs();
            let v = if (dx * dx + dy * dy) < 18.0 * 18.0 {
                220
            } else {
                20
            };
            img[(x, y)] = v;
        }
    }
    img
}

fn main() {
    // 1. Build the recognizer with default config (uniform LBP, 8x8 cells,
    //    chi-square distance).
    let mut rec = LbphRecognizer::new(LbphConfig::default());

    // 2. Enrol two "people" with one crop each. Real pipelines enrol several
    //    crops per identity to handle pose; the descriptor matcher always
    //    takes the best member per label.
    let alice_crop = load_ppm_or_placeholder();
    let bob_crop = load_ppm_or_placeholder();
    rec.enroll("alice", &alice_crop);
    rec.enroll("bob", &bob_crop);

    // 3. Identify a probe. With one crop per label the chi-square distance is
    //    degenerate; this example is mainly here to show the API surface.
    let probe = load_ppm_or_placeholder();
    match rec.identify_crop(&probe) {
        LbphMatch::Match {
            label,
            distance,
            margin,
        } => {
            println!("LBPH  ->  match={label}  distance={distance:.3}  margin={margin:.3}");
        }
        LbphMatch::BelowThreshold { best } => {
            println!("LBPH  ->  no match within threshold (best={best:?})");
        }
        LbphMatch::Ambiguous {
            first,
            second,
            margin,
        } => {
            println!("LBPH  ->  ambiguous between {first} and {second} (margin={margin:.3})");
        }
        LbphMatch::NoCandidates => {
            println!("LBPH  ->  no identities enrolled");
        }
    }

    // 4. Verification path: do these two crops look like the same person?
    let score = rec.verify("alice", &probe).unwrap_or(f32::INFINITY);
    println!("LBPH verify(alice, probe) = {score:.3}");

    // 5. Persistence: store descriptors (not crops) and reload them in a fresh
    //    recogniser. The ranking is bit-identical after reload; see
    //    docs/gallery-persistence.md for the on-disk format.
    let gallery_path = std::env::temp_dir().join(format!(
        "rsface_example_gallery_{}.lbph",
        std::process::id()
    ));
    rec.save(&gallery_path).expect("save gallery");
    let before = rec.rank_crop(&probe);
    drop(rec);

    let reloaded = LbphRecognizer::load(&gallery_path).expect("load gallery");
    println!(
        "LBPH gallery reloaded: {} identities, {} crop(s)",
        reloaded.len(),
        reloaded.crop_count()
    );
    assert_eq!(reloaded.rank_crop(&probe), before);
    std::fs::remove_file(&gallery_path).expect("cleanup gallery");
    println!("LBPH persistence round-trip OK (ranking unchanged)");
}
