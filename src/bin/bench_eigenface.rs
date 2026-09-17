//! Zero-dependency eigenfaces accuracy benchmark.
//!
//! Same crop set / labels as `bench_lbph`, but PCA models are gallery-specific, so the
//! protocol is strict leave-one-out: for every probe the recogniser is RETRAINED on the
//! other crops, then the probe is projected and scored against every gallery member.
//! Both metrics ([`EigenMetric::Euclidean`] and [`EigenMetric::Mahalanobis`]) are scored
//! with and without histogram equalisation; the variant with the largest same/different
//! margin is written to docs/bench-results-eigenface.md.
//!
//! Pair observations: one directed distance per unordered pair (probe = lower index,
//! model excludes the probe), so unlike LBPH the gallery endpoint is always a training
//! point — an acknowledged, documented optimism in the reported FAR/FRR; rank-1 is the
//! primary headline metric.
//!
//! Usage:
//!   cargo run --release --bin bench_eigenface -- out/lbph/crops

#[path = "bench_common.rs"]
mod bench_common;

use std::fs;
use std::path::PathBuf;

use bench_common::{best_threshold, eer, label_counts, load_crops, pair_accuracy, Stats};
use rsface::eigenface::{EigenMetric, EigenfaceConfig, EigenfaceRecognizer, DEFAULT_MAX_DISTANCE};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out/lbph/crops"));
    if !dir.is_dir() {
        eprintln!(
            "crops directory {} not found.\nRun the prep step first:\n  tools/lbph_prep.sh",
            dir.display()
        );
        std::process::exit(2);
    }

    let crops = load_crops(&dir);
    if crops.len() < 3 {
        eprintln!(
            "only {} crop(s) under {} — eigenfaces LOO needs at least 3",
            crops.len(),
            dir.display()
        );
        std::process::exit(2);
    }
    let by_label = label_counts(&crops);
    println!(
        "loaded {} crops, {} identities",
        crops.len(),
        by_label.len()
    );
    for (label, n) in &by_label {
        println!("  {label:>12}: {n}");
    }

    let configs = [
        ("Euclidean, raw", false, EigenMetric::Euclidean),
        ("Euclidean, equalised", true, EigenMetric::Euclidean),
        ("Mahalanobis, raw", false, EigenMetric::Mahalanobis),
        ("Mahalanobis, equalised", true, EigenMetric::Mahalanobis),
    ];
    let results: Vec<Evaluation> = configs
        .iter()
        .map(|(name, equalize, metric)| {
            let cfg = EigenfaceConfig {
                equalize: *equalize,
                metric: *metric,
                ..EigenfaceConfig::default()
            };
            println!("running LOO: {name} ...");
            Evaluation::run(&crops, cfg, name)
        })
        .collect();

    for r in &results {
        println!("\n{}", r.summary());
    }

    let best = results
        .iter()
        .max_by(|a, b| a.margin().total_cmp(&b.margin()))
        .expect("non-empty configs");
    println!(
        "\nreporting {:?} in docs/bench-results-eigenface.md (margin {:.3})",
        best.name,
        best.margin()
    );
    fs::write(
        "docs/bench-results-eigenface.md",
        best.markdown(&dir, &by_label, &results),
    )
    .expect("write report");
    println!("wrote docs/bench-results-eigenface.md");
}

/// LOO evaluation for one configuration.
struct Evaluation {
    name: &'static str,
    same: Vec<f32>,
    different: Vec<f32>,
    best_threshold: f32,
    best_accuracy: f32,
    eer_threshold: f32,
    loo_rank1_repeated: f32,
    loo_repeated_n: usize,
    loo_singletons: usize,
}

impl Evaluation {
    fn run(crops: &[bench_common::Crop], cfg: EigenfaceConfig, name: &'static str) -> Self {
        let n = crops.len();
        let counts = label_counts(crops);
        let mut same = Vec::new();
        let mut different = Vec::new();
        let mut correct = 0usize;
        let mut repeated = 0usize;
        let mut singletons = 0usize;

        for i in 0..n {
            let gallery: Vec<(&str, &rsface::image::GrayImage)> = crops
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, c)| (c.label.as_str(), &c.img))
                .collect();
            let rec = EigenfaceRecognizer::train(cfg, gallery).expect("LOO train");
            let probe = rec.project(&crops[i].img);

            // Pair distances: one directed observation per unordered pair (j > i).
            for j in (i + 1)..n {
                let other = rec.project(&crops[j].img);
                let d = rec.coefficient_distance(&probe, &other);
                if crops[i].label == crops[j].label {
                    same.push(d);
                } else {
                    different.push(d);
                }
            }

            if counts[&crops[i].label] == 1 {
                singletons += 1;
            } else {
                repeated += 1;
                if let Some((label, _)) = rec.rank(&probe).first() {
                    if label == &crops[i].label {
                        correct += 1;
                    }
                }
            }
        }

        let (best_threshold, best_accuracy) = best_threshold(&same, &different);
        let eer_threshold = eer(&same, &different);
        Self {
            name,
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
        format!(
            "{}\n  same identity      n={} mean={:.3} p5={:.3} p50={:.3} p95={:.3} max={:.3}\n\
             \x20 different identity n={} mean={:.3} p5={:.3} p50={:.3} p95={:.3} min={:.3}\n\
             \x20 margin = {:.3}\n  best pair threshold = {:.3} -> pair accuracy {:.1}%\n\
             \x20 EER threshold = {:.3}\n  LOO rank-1 (repeated identities) = {}/{} = {:.1}% \
             (+ {} singletons)",
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
            (self.loo_rank1_repeated * self.loo_repeated_n as f32) as usize,
            self.loo_repeated_n,
            self.loo_rank1_repeated * 100.0,
            self.loo_singletons,
        )
    }

    fn markdown(
        &self,
        dir: &std::path::Path,
        labels: &std::collections::BTreeMap<String, usize>,
        all: &[Evaluation],
    ) -> String {
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
        let (far_def, frr_def) = far_frr(DEFAULT_MAX_DISTANCE);
        let at_default = pair_accuracy(DEFAULT_MAX_DISTANCE, &self.same, &self.different);

        let mut md = String::new();
        md.push_str("# Eigenfaces zero-dependency recogniser — measured accuracy\n\n");
        md.push_str(&format!(
            "crops: `{}` ({} identities, {} crops)\n\n",
            dir.display(),
            labels.len(),
            self.loo_repeated_n + self.loo_singletons
        ));
        md.push_str("## Protocol\n\n");
        md.push_str(
            "Strict leave-one-out: for each probe the PCA model is **retrained** on the \
             other n−1 crops (Jacobi eigendecomposition of the Gram matrix, 64×64 vectors, \
             98 % eigenvalue energy kept). Pair distances are directed (probe vs gallery \
             member; one observation per unordered pair); the gallery endpoint is a \
             training point, so FAR/FRR below is mildly optimistic compared with a probe \
             on unseen data. **LOO rank-1 is the primary metric** — there the probe itself \
             is never in the model.\n\n",
        );
        md.push_str("## Variant comparison (margin = different-mean − same-mean)\n\n");
        md.push_str(
            "| variant | margin | best-threshold pair acc | EER threshold | LOO rank-1 |\n",
        );
        md.push_str("|---|--:|--:|--:|--:|\n");
        for r in all {
            let bold = if r.name == self.name { "**" } else { "" };
            md.push_str(&format!(
                "| {bold}{}{bold} | {bold}{:.3}{bold} | {bold}{:.1}%{bold} | {:.3} | {bold}{}/{} = {:.1}%{bold} |\n",
                r.name,
                r.margin(),
                r.best_accuracy * 100.0,
                r.eer_threshold,
                (r.loo_rank1_repeated * r.loo_repeated_n as f32) as usize,
                r.loo_repeated_n,
                r.loo_rank1_repeated * 100.0,
            ));
        }
        md.push_str(&format!("\nWinner reported below: **{}**\n\n", self.name));
        md.push_str("## Distance distributions\n\n");
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
        md.push_str(&format!("margin: **{:.3}**\n\n", self.margin()));
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
            "| crate default `DEFAULT_MAX_DISTANCE` | {} | {:.1}% | {:.2}% | {:.2}% |\n\n",
            DEFAULT_MAX_DISTANCE,
            at_default * 100.0,
            far_def * 100.0,
            frr_def * 100.0
        ));
        md.push_str("## Identification\n\n");
        md.push_str(&format!(
            "LOO rank-1 over repeated identities: **{}/{} = {:.1}%**. {} singleton \
             probe(s) reported separately.\n\n",
            (self.loo_rank1_repeated * self.loo_repeated_n as f32) as usize,
            self.loo_repeated_n,
            self.loo_rank1_repeated * 100.0,
            self.loo_singletons
        ));
        md.push_str("## Identities in this run\n\n");
        for (label, count) in labels {
            md.push_str(&format!("- `{label}`: {count} crops\n"));
        }
        md.push_str(
            "\n_Generated by `cargo run --release --bin bench_eigenface`; crops prepared by \
             `prep_lbph_crops` with ArcFace-verified identity labels._\n",
        );
        md
    }
}
