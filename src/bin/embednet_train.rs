//! Train the zero-dependency EmbedNet embedding network on a labelled
//! face-crop directory.
//!
//! Same clippy allow set as src/lib.rs (binary targets don't inherit the
//! lib's crate-level allows).
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::identity_op)]
#![allow(clippy::erasing_op)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_saturating_arithmetic)]
#![allow(unknown_lints)]
#![allow(clippy::manual_checked_ops)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::io_other_error)]
#![allow(clippy::mut_from_ref)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::while_let_loop)]
#![allow(clippy::new_without_default)]
#![allow(clippy::needless_collect)]
#![allow(clippy::unused_self)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::single_match)]
#![allow(clippy::no_effect)]
#![allow(clippy::ptr_arg)]
#![allow(unused_parens)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]
//!
//! # Usage
//!
//! ```text
//! cargo run --release --bin embednet_train -- <DIR> [STEPS] [OUT] [SEED] \
//!     [--loss contrastive|arcface] [--no-aug]
//! ```
//!
//! `<DIR>` is a folder of folders — one subdirectory per identity:
//!
//! ```text
//! crops/
//! ├── alice/   a1.pgm a2.png ...
//! ├── bob/     b1.pgm ...
//! └── carol/   ...
//! ```
//!
//! Supported formats: `.pgm` / `.ppm` / `.png`, all decoded by the
//! crate's own zero-dependency codecs. Every crop is resized to the
//! network's 56×56 input with per-image standardisation, then trained
//! with Adam under one of two losses selected by `--loss`:
//!
//! - `contrastive` (default): balanced same/different pairs with the
//!   pair hinge margin.
//! - `arcface`: the additive angular-margin softmax head trained on
//!   randomly sampled labelled crops (one L2-normalised weight row per
//!   identity); validation reports held-out classification accuracy.
//!
//! Training samples are augmented with a random horizontal flip (a
//! near-free identity-preserving transform); pass `--no-aug` to disable.
//!
//! The best validation snapshot is written as `embednet.rsen` (override
//! with `OUT`), loadable via `EmbedNet::load` and `EmbedNetRecognizer`.

use rsface::embednet::{
    ArcFaceTrainer, ContrastiveTrainer, EmbedNet, Rng, DEFAULT_LR, DEFAULT_MARGIN,
};
use rsface::image::codec::{read_pgm, read_ppm};
use rsface::image::png::decode_to_gray;
use rsface::image::GrayImage;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// One identity's standardised training patches.
struct Identity {
    label: String,
    train: Vec<Vec<f32>>,
    validation: Vec<Vec<f32>>,
}

fn load_crop(path: &Path) -> Option<GrayImage> {
    let bytes = fs::read(path).ok()?;
    let mut cur = Cursor::new(bytes);
    let gray = match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "pgm" => read_pgm(&mut cur).ok()?,
        "ppm" => read_ppm(&mut cur).ok()?.to_gray(),
        "png" => decode_to_gray(&mut cur).ok()?,
        other => {
            eprintln!("[embednet_train] skipping unsupported extension: {other}");
            return None;
        }
    };
    Some(gray)
}

fn load_dataset(root: &Path) -> std::io::Result<Vec<Identity>> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    if dirs.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "need at least 2 identity folders under {}, found {}",
                root.display(),
                dirs.len()
            ),
        ));
    }

    let mut identities = Vec::new();
    for dir in dirs {
        let label = dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let mut files: Vec<PathBuf> = fs::read_dir(&dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file())
            .collect();
        files.sort();
        // Deterministic even/odd train/validation split per identity.
        let mut train = Vec::new();
        let mut validation = Vec::new();
        for (i, f) in files.iter().enumerate() {
            let Some(crop) = load_crop(f) else {
                eprintln!(
                    "[embednet_train] could not decode {}, skipping",
                    f.display()
                );
                continue;
            };
            let patch = EmbedNet::preprocess(&crop);
            if i % 2 == 0 {
                train.push(patch);
            } else {
                validation.push(patch);
            }
        }
        // Single-shot identities contribute their only crop to both sets.
        if validation.is_empty() && !train.is_empty() {
            validation.push(train[0].clone());
        }
        if train.is_empty() {
            eprintln!("[embednet_train] identity '{label}' has no usable crops, skipping");
            continue;
        }
        println!(
            "[embednet_train] {label}: {} train / {} val crop(s)",
            train.len(),
            validation.len()
        );
        identities.push(Identity {
            label,
            train,
            validation,
        });
    }
    if identities.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "fewer than 2 identities have usable crops",
        ));
    }
    Ok(identities)
}

/// Sample a balanced pair; returns (a, b, same).
fn sample_pair<'a>(
    ids: &'a [Identity],
    rng: &mut Rng,
    validation: bool,
) -> (&'a [f32], &'a [f32], bool) {
    let pick = |id: &'a Identity, rng: &mut Rng| -> &'a [f32] {
        let pool = if validation {
            &id.validation
        } else {
            &id.train
        };
        let i = (rng.next_u64() as usize) % pool.len();
        &pool[i]
    };
    let same = rng.uniform() < 0.5;
    let i = (rng.next_u64() as usize) % ids.len();
    let a = pick(&ids[i], rng);
    if same {
        (a, pick(&ids[i], rng), true)
    } else {
        let mut j = (rng.next_u64() as usize) % ids.len();
        if j == i {
            j = (j + 1) % ids.len();
        }
        (a, pick(&ids[j], rng), false)
    }
}

fn validate(net: &EmbedNet, ids: &[Identity], pairs: usize, rng: &mut Rng) -> (f32, f32, f32) {
    let threshold = DEFAULT_MARGIN * 0.5;
    let mut correct = 0u32;
    let (mut same_sum, mut same_n) = (0.0f32, 0u32);
    let (mut diff_sum, mut diff_n) = (0.0f32, 0u32);
    for _ in 0..pairs {
        let (a, b, same) = sample_pair(ids, rng, true);
        let ea = match rsface::embedding::Embedding::from_raw(&net.forward_raw(a)) {
            Some(e) => e,
            None => continue,
        };
        let eb = match rsface::embedding::Embedding::from_raw(&net.forward_raw(b)) {
            Some(e) => e,
            None => continue,
        };
        let d = ea.sq_euclidean(&eb).unwrap().sqrt();
        if (d < threshold) == same {
            correct += 1;
        }
        if same {
            same_sum += d;
            same_n += 1;
        } else {
            diff_sum += d;
            diff_n += 1;
        }
    }
    (
        correct as f32 / pairs as f32,
        same_sum / same_n.max(1) as f32,
        diff_sum / diff_n.max(1) as f32,
    )
}

fn main() {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let mut loss = LossKind::Contrastive;
    let mut aug = true;
    let mut positionals: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < raw_args.len() {
        let arg = raw_args[i].clone();
        match arg.as_str() {
            "--loss" => {
                i += 1;
                match raw_args.get(i).map(String::as_str) {
                    Some("contrastive") => loss = LossKind::Contrastive,
                    Some("arcface") => loss = LossKind::ArcFace,
                    Some(value) => arg_error(&format!("unknown --loss value '{value}'")),
                    None => arg_error("--loss requires a value (contrastive|arcface)"),
                }
            }
            "--no-aug" => aug = false,
            value if value.starts_with("--loss=") => match value.trim_start_matches("--loss=") {
                "contrastive" => loss = LossKind::Contrastive,
                "arcface" => loss = LossKind::ArcFace,
                other => arg_error(&format!("unknown --loss value '{other}'")),
            },
            value if value.starts_with("--") => arg_error(&format!("unknown flag '{value}'")),
            _ => positionals.push(arg),
        }
        i += 1;
    }

    let Some(dir) = positionals.first() else {
        eprintln!(
            "usage: embednet_train <DIR> [STEPS] [OUT] [SEED] \
             [--loss contrastive|arcface] [--no-aug]"
        );
        std::process::exit(2);
    };
    let steps: usize = positionals
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);
    let out = positionals
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("embednet.rsen"));
    let seed: u64 = positionals
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20260920);

    let ids = match load_dataset(Path::new(dir)) {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("[embednet_train] dataset error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "[embednet_train] {n} identities, {steps} steps, loss={loss:?}, aug={aug}, lr={DEFAULT_LR}, seed={seed}",
        n = ids.len()
    );

    let run_result = match loss {
        LossKind::Contrastive => run_contrastive(&ids, steps, &out, seed, aug),
        LossKind::ArcFace => run_arcface(&ids, steps, &out, seed, aug),
    };
    if let Err(e) = run_result {
        eprintln!("[embednet_train] {e}");
        std::process::exit(1);
    }
}

fn arg_error(message: &str) -> ! {
    eprintln!("[embednet_train] {message}");
    std::process::exit(2);
}

/// Training loss selected by `--loss`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LossKind {
    Contrastive,
    ArcFace,
}

/// Mirror a square patch horizontally (left/right column reversal).
fn hflip(patch: &[f32], edge: usize) -> Vec<f32> {
    debug_assert_eq!(patch.len(), edge * edge);
    let mut out = vec![0.0f32; patch.len()];
    for y in 0..edge {
        for x in 0..edge {
            out[y * edge + x] = patch[y * edge + (edge - 1 - x)];
        }
    }
    out
}

/// Apply per-sample augmentation; a random horizontal flip is the cheap
/// identity-preserving default, disabled by `--no-aug`.
fn augment(patch: &[f32], rng: &mut Rng, aug: bool) -> Vec<f32> {
    if aug && rng.uniform() < 0.5 {
        hflip(patch, rsface::embednet::INPUT)
    } else {
        patch.to_vec()
    }
}

/// Train end to end with the contrastive pair hinge. The trainer and its
/// Adam state persist for the whole run.
fn run_contrastive(
    ids: &[Identity],
    steps: usize,
    out: &Path,
    seed: u64,
    aug: bool,
) -> std::io::Result<()> {
    let mut trainer = ContrastiveTrainer::new(seed);
    let mut rng = Rng::new(seed ^ 0x517A_C0DE);
    let mut val_rng = Rng::new(seed ^ 0x1234_ABCD);
    let start = Instant::now();
    let mut best_acc = -1.0f32;
    let mut best = trainer.snapshot();

    for step in 1..=steps {
        let (ra, rb, same) = sample_pair(ids, &mut rng, false);
        let a = augment(ra, &mut rng, aug);
        let b = augment(rb, &mut rng, aug);
        let loss = trainer.train_step(&a, &b, same);
        if step % 200 == 0 || step == 1 {
            let (acc, md, xd) = validate(trainer.net(), ids, 200, &mut val_rng);
            println!(
                "[embednet_train] step={step:>5} loss={loss:.4} val_pair_acc={acc:.3} \
                 D(same)={md:.3} D(diff)={xd:.3}"
            );
            if acc > best_acc {
                best_acc = acc;
                best = trainer.snapshot();
            }
        }
    }

    best.save(out)?;
    println!(
        "[embednet_train] saved best (val_pair_acc={best_acc:.3}) to {} in {:.1}s",
        out.display(),
        start.elapsed().as_secs_f32()
    );
    Ok(())
}

/// Train end to end with the ArcFace angular-margin softmax head.
fn run_arcface(
    ids: &[Identity],
    steps: usize,
    out: &Path,
    seed: u64,
    aug: bool,
) -> std::io::Result<()> {
    let mut trainer = ArcFaceTrainer::new(seed, ids.len());
    let mut rng = Rng::new(seed ^ 0x517A_C0DE);
    let start = Instant::now();
    let mut best_acc = -1.0f32;
    let mut best = trainer.snapshot();

    for step in 1..=steps {
        let label = (rng.next_u64() as usize) % ids.len();
        let pool = &ids[label].train;
        let index = (rng.next_u64() as usize) % pool.len();
        let sample = augment(&pool[index], &mut rng, aug);
        let loss = trainer.train_step(&sample, label);
        if step % 200 == 0 || step == 1 {
            let acc = validate_arcface(&trainer, ids);
            println!("[embednet_train] step={step:>5} loss={loss:.4} val_cls_acc={acc:.3}");
            if acc > best_acc {
                best_acc = acc;
                best = trainer.snapshot();
            }
        }
    }

    best.save(out)?;
    println!(
        "[embednet_train] saved best (val_cls_acc={best_acc:.3}) to {} in {:.1}s",
        out.display(),
        start.elapsed().as_secs_f32()
    );
    Ok(())
}

/// Held-out classification accuracy: every validation crop is classified
/// by plain cosine similarity against the head, without augmentation.
fn validate_arcface(trainer: &ArcFaceTrainer, ids: &[Identity]) -> f32 {
    let (mut correct, mut total) = (0u32, 0u32);
    for (label, id) in ids.iter().enumerate() {
        for patch in &id.validation {
            if trainer.classify(patch) == Some(label) {
                correct += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    correct as f32 / total as f32
}
