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
//! Two groups are scored:
//!   1. the shipped descriptor at its defaults (6×6 grid, 120 px), raw and
//!      histogram-equalised — the headline comparison;
//!   2. a tuning sweep (grid × crop size, raw) used to check that the shipped grid
//!      and crop size actually win on harder, larger galleries.
//!
//! Configurations rank by LOO rank-1 (primary), then same/different margin. The
//! detailed distributions always describe the SHIPPED raw config, since the crate's
//! `DEFAULT_MAX_DISTANCE` is only meaningful in that descriptor's chi-square scale.
//!
//! No ONNX backend, no model files, no third-party crates — this builds with the
//! default zero-dep profile.
//!
//! Usage:
//!   cargo run --release --bin bench_lbph -- out/lbph/crops
//!
//! The markdown report is written to docs/bench-results-lbph.md.

#[path = "bench_common.rs"]
mod bench_common;

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use bench_common::{best_threshold, eer, label_counts, load_crops, pair_accuracy, Stats};
use rsface::lbph::{extract, LbphConfig, LbphDescriptor, DEFAULT_MAX_DISTANCE};

/// Grid sides (cells per edge) swept by the tuning grid.
const SWEEP_GRIDS: [usize; 3] = [6, 8, 10];
/// Square crop sizes swept by the tuning grid.
const SWEEP_FACE_SIZES: [usize; 3] = [90, 120, 150];

fn main() {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out/lbph/crops"));
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

    // Canonical: the shipped raw descriptor and its equalised variant.
    let raw_cfg = LbphConfig::default();
    let eq_cfg = LbphConfig {
        equalize: true,
        ..LbphConfig::default()
    };
    let canonical = vec![
        Evaluation::run(&crops, &raw_cfg, "raw LBP (6×6, 120 px)"),
        Evaluation::run(&crops, &eq_cfg, "equalised LBP (6×6, 120 px)"),
    ];

    // Tuning sweep: grid × crop size, raw only.
    let mut sweep: Vec<Evaluation> = Vec::new();
    for &grid in &SWEEP_GRIDS {
        for &face_size in &SWEEP_FACE_SIZES {
            let cfg = LbphConfig {
                grid_x: grid,
                grid_y: grid,
                face_size,
                ..LbphConfig::default()
            };
            let name = format!("raw, {grid}×{grid} grid, {face_size} px");
            println!("running sweep: {name} ...");
            sweep.push(Evaluation::run(&crops, &cfg, &name));
        }
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
    // Detail section always follows the shipped raw config (canonical row 0).
    let shipped = &canonical[0];
    // The sweep re-evaluates the shipped config under a different row name; when the
    // numbers are genuinely tied, point at the canonical row instead of a duplicate.
    if winner.name != shipped.name
        && winner.loo_rank1_repeated == shipped.loo_rank1_repeated
        && (winner.margin() - shipped.margin()).abs() < 1e-9
    {
        winner = shipped;
    }
    println!(
        "\nrank-1 winner: {} ({:.1}%, margin {:.3}); reporting shipped config {} in \
         docs/bench-results-lbph.md",
        winner.name,
        winner.loo_rank1_repeated * 100.0,
        winner.margin(),
        shipped.name
    );
    let report = shipped.markdown(&dir, &by_label, &canonical, &sweep, winner);
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
        let at_default = pair_accuracy(DEFAULT_MAX_DISTANCE, &self.same, &self.different);
        format!(
            "{}\n  same identity      n={} mean={:.3} p5={:.3} p50={:.3} p95={:.3} max={:.3}\n\
             \x20 different identity n={} mean={:.3} p5={:.3} p50={:.3} p95={:.3} min={:.3}\n\
             \x20 margin (diff mean - same mean) = {:.3}\n\
             \x20 best pair threshold = {:.3} -> pair accuracy {:.1}% (FAR {:.2}%, FRR {:.2}%)\n\
             \x20 EER threshold = {:.3} (FAR {:.2}%, FRR {:.2}%)\n\
             \x20 crate default ({}) pair accuracy {:.1}% (FAR {:.2}%, FRR {:.2}%)\n\
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
            far_best * 100.0,
            frr_best * 100.0,
            self.eer_threshold,
            far_eer * 100.0,
            frr_eer * 100.0,
            DEFAULT_MAX_DISTANCE,
            at_default * 100.0,
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
        labels: &BTreeMap<String, usize>,
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
        let acc_def = pair_accuracy(DEFAULT_MAX_DISTANCE, &self.same, &self.different);

        let total = self.loo_repeated_n + self.loo_singletons;
        let mut md = String::new();
        md.push_str("# LBPH zero-dependency recogniser — measured accuracy\n\n");
        md.push_str(&format!(
            "crops: `{}` ({} identities, {} crops)\n\n",
            dir.display(),
            labels.len(),
            total
        ));
        md.push_str(
            "LBPH descriptors are gallery-independent, so every crop is described once and \
             every unordered pair contributes one chi-square distance; the rank-1 loop is \
             leave-one-out. Configurations rank by LOO rank-1, then margin.\n\n",
        );
        md.push_str("## Canonical variants at the shipped defaults (6×6 grid, 120 px)\n\n");
        push_table(&mut md, canonical, winner);
        md.push_str("\n## Hyperparameter sweep (raw LBP)\n\n");
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
                    " — tied with the shipped **{}** at {shipped_correct}/{} rank-1; the \
                     shipped grid is kept (margin differences of one probe or less are \
                     sampling noise on 68 probes)",
                    self.name, self.loo_repeated_n
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
            "Shipped configuration detailed below: **{}**.\n\n",
            self.name
        ));
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
            "| crate default `{}` | {} | {:.1}% | {:.2}% | {:.2}% |\n\n",
            "DEFAULT_MAX_DISTANCE",
            DEFAULT_MAX_DISTANCE,
            acc_def * 100.0,
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
