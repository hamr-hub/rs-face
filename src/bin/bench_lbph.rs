//! Zero-dependency LBPH accuracy benchmark.
//!
//! Reads a directory tree of small grayscale face crops prepared by
//! `prep_lbph_crops` (or any `*.pgm` / `*.ppm` / `*.png` tree whose filenames are
//! `<identity>__<source>__<frame>.*` — the part before the first `__` is the ground
//! truth label) and reports:
//!
//!   * chi-square distance distributions for same-identity vs different-identity pairs
//!   * best operating threshold, accuracy at that threshold, EER, and FAR/FRR at the
//!     crate default
//!   * leave-one-out rank-1 identification rate
//!
//! Both the raw and histogram-equalised descriptors are scored. No ONNX backend, no
//! model files, no third-party crates — this builds with the default zero-dep profile.
//!
//! Usage:
//!   cargo run --release --bin bench_lbph -- out/lbph_crops
//!
//! The markdown report is written to docs/bench-results-lbph.md.

#[path = "bench_common.rs"]
mod bench_common;

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use bench_common::{best_threshold, eer, label_counts, load_crops, pair_accuracy, Stats};
use rsface::lbph::{extract, LbphConfig, LbphDescriptor};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out/lbph_crops"));
    if !dir.is_dir() {
        eprintln!(
            "crops directory {} not found.\nRun the prep step first:\n  \
             tools/lbph_prep.sh",
            dir.display()
        );
        std::process::exit(2);
    }

    let crops = load_crops(&dir);
    if crops.len() < 2 {
        eprintln!(
            "only {} crop(s) under {} — need at least 2",
            crops.len(),
            dir.display()
        );
        std::process::exit(2);
    }

    println!("loaded {} crops:", crops.len());
    let by_label = label_counts(&crops);
    for (label, n) in &by_label {
        println!("  {label:>12}: {n}");
    }

    let raw_cfg = LbphConfig::default();
    let eq_cfg = LbphConfig {
        equalize: true,
        ..LbphConfig::default()
    };

    let raw = Evaluation::run(&crops, &raw_cfg, "raw LBP");
    let eq = Evaluation::run(&crops, &eq_cfg, "histogram-equalised LBP");

    println!("\n{}", raw.summary());
    println!("\n{}", eq.summary());

    // Pick whichever preprocessing separates identities better and persist it.
    let best = if raw.margin() >= eq.margin() {
        &raw
    } else {
        &eq
    };
    println!(
        "\nreporting {:?} preprocessing in docs/bench-results-lbph.md (margin {:.3})",
        best.name,
        best.margin()
    );
    let report = best.markdown(&dir, &by_label);
    fs::write("docs/bench-results-lbph.md", report).expect("write report");
    println!("wrote docs/bench-results-lbph.md");
}

/// Aggregated evaluation for one configuration.
struct Evaluation {
    name: String,
    same: Vec<f32>,
    different: Vec<f32>,
    /// (threshold, pair-classification accuracy)
    best_threshold: f32,
    best_accuracy: f32,
    eer_threshold: f32,
    /// Fraction of repeated-identity probes whose nearest crop shares their label.
    loo_rank1_repeated: f32,
    loo_repeated_n: usize,
    /// Singleton-identity probes — rank-1 can never succeed for these.
    loo_singletons: usize,
}

impl Evaluation {
    fn run(crops: &[bench_common::Crop], cfg: &LbphConfig, name: &str) -> Self {
        let descs: Vec<LbphDescriptor> = crops.iter().map(|c| extract(&c.img, cfg)).collect();
        let n = descs.len();

        let mut same = Vec::new();
        let mut different = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = descs[i].chi_square(&descs[j]).expect("same grid shape");
                if crops[i].label == crops[j].label {
                    same.push(d);
                } else {
                    different.push(d);
                }
            }
        }

        let (best_threshold, best_accuracy) = best_threshold(&same, &different);
        let eer_threshold = eer(&same, &different);

        // Leave-one-out rank-1: every crop is a probe against all OTHER crops;
        // the single nearest descriptor decides the predicted identity. Probes whose
        // identity appears only once have no right answer in the gallery, so report
        // their count separately instead of counting them as identification errors.
        let counts = label_counts(crops);
        let mut correct = 0usize;
        let mut repeated = 0usize;
        let mut singletons = 0usize;
        for i in 0..n {
            let mut best: Option<(f32, &str)> = None;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = descs[i].chi_square(&descs[j]).unwrap();
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, crops[j].label.as_str()));
                }
            }
            if counts[&crops[i].label] == 1 {
                singletons += 1;
            } else {
                repeated += 1;
                if best.is_some_and(|(_, label)| label == crops[i].label) {
                    correct += 1;
                }
            }
        }

        Self {
            name: name.to_string(),
            same,
            different,
            best_threshold,
            best_accuracy,
            eer_threshold,
            loo_rank1_repeated: correct as f32 / repeated.max(1) as f32,
            loo_repeated_n: repeated,
            loo_singletons: singletons,
        }
    }

    fn margin(&self) -> f32 {
        Stats::of(&self.different).mean - Stats::of(&self.same).mean
    }

    fn summary(&self) -> String {
        let s = Stats::of(&self.same);
        let d = Stats::of(&self.different);
        let at_default = pair_accuracy(
            rsface::lbph::DEFAULT_MAX_DISTANCE,
            &self.same,
            &self.different,
        );
        format!(
            "{}\n  same identity      n={} mean={:.3} p5={:.3} p50={:.3} p95={:.3} max={:.3}\n\
             \x20 different identity n={} mean={:.3} p5={:.3} p50={:.3} p95={:.3} min={:.3}\n\
             \x20 margin (diff mean - same mean) = {:.3}\n\
             \x20 best pair threshold = {:.3} -> pair accuracy {:.1}%\n\
             \x20 EER threshold = {:.3}\n\
             \x20 pair accuracy at crate default ({}) = {:.1}%\n\
             \x20 leave-one-out rank-1 (repeated identities) = {}/{} = {:.1}% \
             (+ {} singleton probes without a gallery match)",
            self.name,
            self.same.len(),
            s.mean,
            s.p05,
            s.p50,
            s.p95,
            s.max,
            self.different.len(),
            d.mean,
            d.p05,
            d.p50,
            d.p95,
            d.min,
            self.margin(),
            self.best_threshold,
            self.best_accuracy * 100.0,
            self.eer_threshold,
            rsface::lbph::DEFAULT_MAX_DISTANCE,
            at_default * 100.0,
            (self.loo_rank1_repeated * self.loo_repeated_n as f32) as usize,
            self.loo_repeated_n,
            self.loo_rank1_repeated * 100.0,
            self.loo_singletons,
        )
    }

    fn markdown(&self, dir: &std::path::Path, labels: &BTreeMap<String, usize>) -> String {
        let s = Stats::of(&self.same);
        let d = Stats::of(&self.different);
        let far_frr = |t: f32| -> (f32, f32) {
            let frr = self.same.iter().filter(|&&x| x > t).count() as f32 / self.same.len() as f32;
            let far = self.different.iter().filter(|&&x| x <= t).count() as f32
                / self.different.len() as f32;
            (far, frr)
        };
        let (far_best, frr_best) = far_frr(self.best_threshold);
        let (far_eer, frr_eer) = far_frr(self.eer_threshold);
        let (far_def, frr_def) = far_frr(rsface::lbph::DEFAULT_MAX_DISTANCE);

        let total = self.loo_repeated_n + self.loo_singletons;
        let mut md = String::new();
        md.push_str("# LBPH zero-dependency recogniser — measured accuracy\n\n");
        md.push_str(&format!(
            "crops: `{}` ({} identities, {} crops)\n\n",
            dir.display(),
            labels.len(),
            total
        ));
        md.push_str(&format!("preprocessing: **{}**\n\n", self.name));
        md.push_str("## Chi-square distance distributions\n\n");
        md.push_str("| pair type | n | mean | p5 | p50 | p95 | min/max |\n");
        md.push_str("|---|--:|--:|--:|--:|--:|--:|\n");
        md.push_str(&format!(
            "| same identity | {} | {:.3} | {:.3} | {:.3} | {:.3} | max {:.3} |\n",
            self.same.len(),
            s.mean,
            s.p05,
            s.p50,
            s.p95,
            s.max
        ));
        md.push_str(&format!(
            "| different identity | {} | {:.3} | {:.3} | {:.3} | {:.3} | min {:.3} |\n\n",
            self.different.len(),
            d.mean,
            d.p05,
            d.p50,
            d.p95,
            d.min
        ));
        md.push_str(&format!(
            "margin (different mean − same mean): **{:.3}**\n\n",
            self.margin()
        ));
        md.push_str("## Operating points\n\n");
        md.push_str("| point | threshold | pair accuracy | FAR | FRR |\n");
        md.push_str("|---|--:|--:|--:|--:|\n");
        md.push_str(&format!(
            "| best pair accuracy | {:.3} | {:.1}% | {:.2}% | {:.2}% |\n",
            self.best_threshold,
            self.best_accuracy * 100.0,
            far_best * 100.0,
            frr_best * 100.0
        ));
        md.push_str(&format!(
            "| EER | {:.3} | — | {:.2}% | {:.2}% |\n",
            self.eer_threshold,
            far_eer * 100.0,
            frr_eer * 100.0
        ));
        md.push_str(&format!(
            "| crate default `{}` | {} | — | {:.2}% | {:.2}% |\n\n",
            "DEFAULT_MAX_DISTANCE",
            rsface::lbph::DEFAULT_MAX_DISTANCE,
            far_def * 100.0,
            frr_def * 100.0
        ));
        md.push_str("## Identification\n\n");
        md.push_str(&format!(
            "Leave-one-out rank-1 over repeated identities (each crop probed against every \
             other crop of the same gallery): **{}/{} = {:.1}%**.\n\n\
             {} crop(s) belong to singleton identities with no same-identity gallery sample; \
             rank-1 cannot succeed for those and they are reported separately rather than \
             counted as errors.\n\n",
            (self.loo_rank1_repeated * self.loo_repeated_n as f32) as usize,
            self.loo_repeated_n,
            self.loo_rank1_repeated * 100.0,
            self.loo_singletons,
        ));
        md.push_str("## Identities in this run\n\n");
        for (label, n) in labels {
            md.push_str(&format!("- `{label}`: {n} crops\n"));
        }
        md.push_str("\n_Generated by `cargo run --release --bin bench_lbph`; crops prepared by ");
        md.push_str("`prep_lbph_crops` with ArcFace-verified identity labels._\n");
        md
    }
}
