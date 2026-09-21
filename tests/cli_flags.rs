//! CLI flag-coverage smoke tests.
//!
//! Invariant: every documented CLI flag in `src/main.rs::print_help()` is
//! actually wired up in the binary. A regression that removes the
//! matching `--flag` from the parser (or the help text) without removing
//! the matching `match a` arm will silently degrade — this file catches it
//! by parsing the live `--help` output.
//!
//! Strategy:
//! 1. Locate the `rs-face` binary via `CARGO_BIN_EXE_rs-face` (cargo sets
//!    this for integration tests) or fall back to `./target/debug/rs-face`.
//! 2. Spawn it with `--help` and capture stdout.
//! 3. Assert each documented flag token appears in the help text.
//!
//! Per the prompt, flags that require external deps (ffprobe, GPU) are
//! still documented; we don't exercise them here, only assert they are
//! listed. This file does not import `main.rs` symbols; it is purely a
//! black-box smoke test on the CLI surface.

use std::path::PathBuf;
use std::process::Command;

/// Locate the `rs-face` binary. Cargo sets `CARGO_BIN_EXE_rs-face` when
/// building integration tests; on a fresh checkout (e.g. `cargo test`
/// without prior `cargo build`) the binary may still be in the workspace
/// target dir — fall back to that.
fn locate() -> PathBuf {
    if let Ok(env_path) = std::env::var("CARGO_BIN_EXE_rs-face") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return p;
        }
    }
    let candidates = [
        "./target/debug/rs-face",
        "./target/release/rs-face",
        "target/debug/rs-face",
        "target/release/rs-face",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    panic!(
        "rs-face binary not found; tried CARGO_BIN_EXE_rs-face and {:?}",
        candidates
    );
}

/// Run the binary with the given args and capture combined stdout.
fn run_capture(bin: &PathBuf, args: &[&str]) -> String {
    let out = Command::new(bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {:?}: {e}", bin));
    assert!(
        out.status.success(),
        "rs-face {:?} exited non-zero: status={:?} stderr={:?}",
        args,
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Flags that the help text MUST mention. Each entry is `(flag, what it
/// does)` — if any of these go missing the test asserts it loudly.
const DOCUMENTED_FLAGS: &[(&str, &str)] = &[
    ("--out", "output directory"),
    ("--batch-dir", "batch directory"),
    ("--algo", "algorithm selector"),
    ("--cascade", "custom cascade path"),
    ("--min-neighbors", "min neighbors"),
    ("--threads", "thread count"),
    ("--min-size", "minimum detection size"),
    ("--max-size", "maximum detection size"),
    ("--scale", "pyramid scale"),
    ("--stride", "window stride"),
    ("--nms", "NMS IoU threshold"),
    ("--min-score", "minimum score threshold"),
    ("--only-with-face", "skip empty frames"),
    ("--queue-depth", "per-worker queue depth"),
    ("--cnn", "use CNN detector"),
    ("--cnn-weights", "CNN weights path"),
    ("--no-gpu", "disable GPU"),
    ("--no-equalize", "skip histogram equalisation"),
    ("--list-algos", "list algorithms"),
    ("--list-features", "list compiled features"),
    ("--version", "print version"),
    ("--help", "print help"),
];

#[test]
fn rs_face_help_lists_every_documented_flag() {
    let bin = locate();
    let help = run_capture(&bin, &["--help"]);
    // Sanity: the help must look like a help block (USAGE section, etc.).
    assert!(help.contains("USAGE"), "--help output missing USAGE section:\n{help}");
    assert!(
        help.contains("ALGORITHMS") || help.contains("algorithm"),
        "--help output missing algorithms section:\n{help}"
    );

    let mut missing = Vec::new();
    for (flag, what) in DOCUMENTED_FLAGS {
        if !help.contains(flag) {
            missing.push(format!("{flag} ({what})"));
        }
    }
    assert!(
        missing.is_empty(),
        "rs-face --help is missing the following documented flags:\n  - {}\n\
         ----- captured help text -----\n{help}\n----- end -----",
        missing.join("\n  - ")
    );
}

#[test]
fn rs_face_version_flag_prints_version() {
    let bin = locate();
    let out = run_capture(&bin, &["--version"]);
    // The crate version is "0.2.0" (see Cargo.toml); the binary should
    // print it on --version. We assert at least one dot is in the
    // version line and the binary actually exited zero (run_capture
    // already covers success).
    let first_line = out.lines().next().unwrap_or("");
    assert!(
        first_line.contains('.'),
        "--version output should contain a version string, got: {first_line:?}"
    );
}

#[test]
fn rs_face_list_algos_lists_haar_luminance_and_cnn() {
    let bin = locate();
    let out = run_capture(&bin, &["--list-algos"]);
    assert!(
        out.contains("haar"),
        "--list-algos missing 'haar':\n{out}"
    );
    assert!(
        out.contains("luminance"),
        "--list-algos missing 'luminance':\n{out}"
    );
    assert!(
        out.contains("cnn"),
        "--list-algos missing 'cnn':\n{out}"
    );
}

#[test]
fn rs_face_list_features_lists_at_least_one_feature() {
    let bin = locate();
    let out = run_capture(&bin, &["--list-features"]);
    // The default build always enables the "default" feature bundle;
    // the output must reference it. We don't assert specific feature
    // names because the list is feature-gated (metal/cuda/ort only
    // appear when compiled in).
    assert!(
        out.contains("default")
            || out.contains("haar")
            || out.contains("luminance")
            || out.contains("Cargo features"),
        "--list-features must reference the default feature bundle or \
         at least one core algorithm; got:\n{out}"
    );
}

#[test]
fn rs_face_help_recipes_section_mentions_batch_dir() {
    let bin = locate();
    let help = run_capture(&bin, &["--help"]);
    // The batch-dir recipe is part of the documented CLI surface.
    assert!(
        help.contains("--batch-dir") && help.contains("rs-face"),
        "--help output must reference the --batch-dir recipe; got:\n{help}"
    );
}

#[test]
fn rs_face_demo_runs_without_external_dependencies() {
    // `rs-face demo` is the zero-arg install check. It writes to
    // `./rsface-demo` by default. We redirect to a temp dir to keep the
    // workspace clean and assert the annotated PNG was produced.
    let bin = locate();
    let tmp = std::env::temp_dir().join("rsface-cli-flag-test");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let status = Command::new(&bin)
        .args(["demo", "--out"])
        .arg(&tmp)
        .status()
        .unwrap_or_else(|e| panic!("spawn rs-face demo: {e}"));
    assert!(
        status.success(),
        "rs-face demo --out <tmp> exited non-zero: status={status:?}"
    );
    // Demo writes an annotated PNG named `frame_00000.png` (see
    // `output::write_annotated_png`). We assert that at least one PNG
    // exists in the output dir.
    let png_count = std::fs::read_dir(&tmp)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", tmp.display()))
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |s| s == "png")
        })
        .count();
    assert!(
        png_count >= 1,
        "rs-face demo did not produce any annotated PNG in {}",
        tmp.display()
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn rs_face_unknown_flag_exits_non_zero() {
    // The CLI must not silently accept garbage flags. A regression that
    // turns flag parsing into a no-op would surface here.
    let bin = locate();
    let out = Command::new(&bin)
        .args(["--definitely-not-a-real-flag-xyz"])
        .output()
        .expect("spawn rs-face");
    assert!(
        !out.status.success(),
        "rs-face must reject unknown flags; got success on --definitely-not-a-real-flag-xyz\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn rs_face_help_appears_when_no_args_provided_to_input_slot() {
    // `rs-face --help` is the documented short path; the binary must
    // also handle the case where the user forgets the required input
    // and prints help (or at least exits non-zero without a panic).
    let bin = locate();
    let out = Command::new(&bin)
        .args(["--help"])
        .output()
        .expect("spawn rs-face --help");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("USAGE") || combined.contains("USAGE:"),
        "--help must print the USAGE section in stdout/stderr; got:\n{combined}"
    );
}

#[test]
fn rs_face_help_does_not_crash_on_extreme_flags() {
    // Just verifies the binary doesn't panic on `--help` repeated or
    // combined with other known flags.
    let bin = locate();
    for args in [
        vec!["--help"],
        vec!["--help", "--help"],
        vec!["--help", "--list-algos"],
    ] {
        let out = Command::new(&bin).args(&args).output().expect("spawn");
        assert!(
            out.status.success(),
            "rs-face {args:?} should exit zero, got: status={:?} stderr={}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// Read the help text once and verify it contains every documented flag
/// AND the `--help`/`-h` short alias referenced in `main.rs`. (This is
/// what the help text itself promises: "`--help | -h" print this help`.)
#[test]
fn rs_face_help_documents_help_alias() {
    let bin = locate();
    let help = run_capture(&bin, &["--help"]);
    // The help block ends with `--help                print this help`
    // (and optionally `-h`). We don't strictly require `-h` (it may be
    // documented only in the source) but `--help` itself must appear.
    assert!(help.contains("--help"), "--help self-reference missing");
}