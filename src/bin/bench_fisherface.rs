//! Zero-dependency Fisherfaces (LDA) accuracy benchmark.
//!
//! Same crop set / labels and the same strict leave-one-out protocol as
//! `bench_eigenface`: for every probe the PCA+LDA model is RETRAINED on the other
//! n−1 crops, then the probe is projected and scored against every gallery member.
//!
//! Two groups are scored:
//!   1. the two canonical variants at the crate defaults (64×64 crops, all C−1
//!      discriminant components) — raw and histogram-equalised;
//!   2. a crop-size sweep (32/48/64 px, raw) used to check the shipped default.
//!
//! Configurations rank by LOO rank-1 (primary), then same/different margin.
//!
//! Pair observations: one directed distance per unordered pair (probe = lower index,
//! model excludes the probe), so the gallery endpoint is always a training point — an
//! acknowledged, documented optimism in the reported FAR/FRR; rank-1 is the primary
//! headline metric.
//!
//! Usage:
//!   cargo run --release --bin bench_fisherface -- out/lbph/crops

#[path = "bench_common.rs"]
mod bench_common;

use std::fs;
use std::path::PathBuf;

use bench_common::{best_threshold, eer, label_counts, load_crops, pair_accuracy, Stats};
use rsface::fisherface::{
    FisherfaceConfig, FisherfaceRecognizer, DEFAULT_FACE_SIZE, DEFAULT_MAX_DISTANCE,
};

/// Face sizes swept by the tuning grid.
const SWEEP_FACE_SIZES: [usize; 3] = [32, 48, DEFAULT_FACE_SIZE];

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
    if crops.len() < 4 {
        eprintln!(
            "only {} crop(s) under {} — fisherfaces LOO needs at least 4 (2 identities twice)",
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

    // Canonical variants at the shipped defaults: raw vs histogram equalisation.
    let canonical: [(String, FisherfaceConfig); 2] = [
        ("Fisherfaces, raw".to_string(), cfg(false)),
        ("Fisherfaces, equalised".to_string(), cfg(true)),
    ];
    let canonical: Vec<Evaluation> = canonical
        .iter()
        .map(|(name, c)| {
            println!("running LOO: {name} ...");
            Evaluation::run(&crops, *c, name)
        })
        .collect();

    // Tuning sweep: crop size, raw preprocessing.
    let mut sweep: Vec<Evaluation> = Vec::new();
    for &face_size in &SWEEP_FACE_SIZES {
        let name = format!("raw, {face_size} px");
        let c = FisherfaceConfig {
            face_size,
            ..FisherfaceConfig::default()
        };
        println!("running LOO sweep: {name} ...");
        sweep.push(Evaluation::run(&crops, c, &name));
    }

    for r in canonical.iter().chain(sweep.iter()) {
        println!("\n{}", r.summary());
    }

    // Winner: honest rank-1 first, margin as tiebreak.
    let mut winner: &Evaluation = canonical
        .iter()
        .chain(sweep.iter())
        .max_by(|a, b| {
            a.loo_rank1_repeated
                .total_cmp(&b.loo_rank1_repeated)
                .then(a.margin().total_cmp(&b.margin()))
        })
        .expect("non-empty configs");
    // The sweep re-evaluates the shipped config under another row name; a genuine tie
    // points at the canonical row rather than the duplicate.
    let shipped = &canonical[0];
    if winner.name != shipped.name
        && winner.loo_rank1_repeated == shipped.loo_rank1_repeated
        && (winner.margin() - shipped.margin()).abs() < 1e-9
    {
        winner = shipped;
    }
    println!(
        "\nrank-1 winner: {} ({:.1}%, margin {:.3}); reporting shipped config {} in \
         docs/bench-results-fisherface.md",
        winner.name,
        winner.loo_rank1_repeated * 100.0,
        winner.margin(),
        shipped.name
    );
    fs::write(
        "docs/bench-results-fisherface.md",
        shipped.markdown(&dir, &by_label, &canonical, &sweep, winner),
    )
    .expect("write report");
    println!("wrote docs/bench-results-fisherface.md");
}

fn cfg(equalize: bool) -> FisherfaceConfig {
    FisherfaceConfig {
        equalize,
        ..FisherfaceConfig::default()
    }
}

/// LOO evaluation for one configuration.
struct Evaluation {
    name: String,
    face_size: usize,
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
    fn run(crops: &[bench_common::Crop], cfg: FisherfaceConfig, name: &str) -> Self {
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
            let rec = FisherfaceRecognizer::train(cfg, gallery).expect("LOO train");
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
            name: name.to_string(),
            face_size: cfg.face_size,
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

    fn rank1_fraction(&self) -> f32 {
        self.loo_rank1_repeated
    }

    fn summary(&self) -> String {
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
        format!(
            "{}\n  same identity      n={} mean={:.3} p5={:.3} p50={:.3} p95={:.3} max={:.3}\n\
             \x20 different identity n={} mean={:.3} p5={:.3} p50={:.3} p95={:.3} min={:.3}\n\
             \x20 margin = {:.3}\n  best pair threshold = {:.3} -> pair accuracy {:.1}% \
             (FAR {:.2}%, FRR {:.2}%)\n\
             \x20 EER threshold = {:.3} (FAR {:.2}%, FRR {:.2}%)\n\
             \x20 crate default ({}) FAR {:.2}%, FRR {:.2}%\n\
             \x20 LOO rank-1 (repeated identities) = {}/{} = {:.1}% (+ {} singletons)",
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
            far_best * 100.0,
            frr_best * 100.0,
            self.eer_threshold,
            far_eer * 100.0,
            frr_eer * 100.0,
            DEFAULT_MAX_DISTANCE,
            far_def * 100.0,
            frr_def * 100.0,
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
        canonical: &[Evaluation],
        sweep: &[Evaluation],
        winner: &Evaluation,
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
        md.push_str("# Fisherfaces (LDA) zero-dependency recogniser — measured accuracy\n\n");
        md.push_str(&format!(
            "crops: `{}` ({} identities, {} crops)\n\n",
            dir.display(),
            labels.len(),
            self.loo_repeated_n + self.loo_singletons
        ));
        md.push_str("## Protocol\n\n");
        md.push_str(
            "Strict leave-one-out: for each probe the PCA→LDA model is **retrained** on the \
             other n−1 crops (Gram-matrix PCA reduction to n−C directions, then at most C−1 \
             Fisher directions from the whitened between/within scatter matrices; crop size \
             varies per row). Pair distances are directed (probe vs gallery member; one \
             observation per unordered pair); the gallery endpoint is a training point, so \
             FAR/FRR below is mildly optimistic compared with a probe on unseen data. **LOO \
             rank-1 is the primary metric** — there the probe itself is never in the model. \
             Configurations rank by LOO rank-1, then margin.\n\n",
        );
        md.push_str("## Canonical variants at the shipped defaults (64 px, C−1 components)\n\n");
        push_table(&mut md, canonical, winner);
        md.push_str("\n## Hyperparameter sweep (raw)\n\n");
        push_table(&mut md, sweep, winner);
        md.push_str(&format!("\nRank-1 winner: **{}**", winner.name));
        if winner.name != self.name {
            let delta = ((winner.loo_rank1_repeated - self.loo_rank1_repeated)
                * self.loo_repeated_n as f32)
                .round() as usize;
            let winner_correct =
                (winner.loo_rank1_repeated * winner.loo_repeated_n as f32) as usize;
            let shipped_correct = (self.loo_rank1_repeated * self.loo_repeated_n as f32) as usize;
            if delta == 0 {
                md.push_str(&format!(
                    " — tied with the shipped **{}** at {}/{} rank-1; the 64 px default is \
                     kept (the margin difference is within one probe of sampling noise and the \
                     constant is calibrated in the shipped scale)",
                    self.name, shipped_correct, self.loo_repeated_n
                ));
            } else {
                md.push_str(&format!(
                    " — {winner_correct}/{} rank-1 vs the shipped **{}** {shipped_correct}/{}, \
                     a difference of {delta} probe(s)",
                    winner.loo_repeated_n, self.name, self.loo_repeated_n
                ));
            }
        } else {
            md.push_str(" — the shipped default configuration wins outright");
        }
        md.push_str(".\n\n");
        md.push_str(&format!(
            "Shipped configuration detailed below: **{}** — {}×{} crops, at most C−1 \
             discriminant components.\n\n",
            self.name, self.face_size, self.face_size
        ));
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
            "| crate default `DEFAULT_MAX_DISTANCE` ({}) | {} | {:.1}% | {:.2}% | {:.2}% |\n\n",
            DEFAULT_MAX_DISTANCE,
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
            "\n_Generated by `cargo run --release --bin bench_fisherface`; crops prepared by \
             `prep_lbph_crops` with ArcFace-verified identity labels._\n",
        );
        md
    }
}

/// One comparison table; the winning row is bolded.
fn push_table(out: &mut String, rows: &[Evaluation], winner: &Evaluation) {
    out.push_str("| variant | margin | best-threshold pair acc | EER threshold | LOO rank-1 |\n");
    out.push_str("|---|--:|--:|--:|--:|\n");
    for r in rows {
        let bold = if r.name == winner.name { "**" } else { "" };
        out.push_str(&format!(
            "| {bold}{}{bold} | {bold}{:.3}{bold} | {bold}{:.1}%{bold} | {:.3} | \
             {bold}{}/{} = {:.1}%{bold} |\n",
            r.name,
            r.margin(),
            r.best_accuracy * 100.0,
            r.eer_threshold,
            (r.rank1_fraction() * r.loo_repeated_n as f32) as usize,
            r.loo_repeated_n,
            r.rank1_fraction() * 100.0,
        ));
    }
}
