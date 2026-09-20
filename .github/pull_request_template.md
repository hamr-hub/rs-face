<!--
  Thanks for contributing to rs-face.
  STRUCTURE.md is the contract — read it first. CI runs on every PR:
  fmt --check · clippy -D warnings · zero-dep build · tests (both crates).
-->
## What

<!-- one-line summary of the change -->

## Why

<!-- problem / motivation; link the issue (Closes #123) -->

## Type of change

- [ ] `fix` — bug fix
- [ ] `feat` — new feature
- [ ] `perf` — performance
- [ ] `refactor` / `docs` / `style` / `test` / `build` / `ci` / `chore`

## Verification

<!-- real evidence, not "should work". paste command + result -->
- [ ] `cargo fmt --all --check` passes
- [ ] `cargo clippy --release --all-targets -- -D warnings` passes
- [ ] `cargo build --no-default-features --lib` passes (zero-dep intact)
- [ ] `cargo test --lib` passes
- [ ] platform: `cargo test --manifest-path platform/Cargo.toml` passes
- [ ] If platform server behavior changed: `bash platform/scripts/docker-smoke.sh`

## Accuracy / correctness (delete if not algorithmic)

- Does this change detection / recognition thresholds, NMS, or a model? 
- What cascade / model / images were used to measure?
- Before/after numbers (false positives, miss rate, latency)?

## Changelog

- [ ] `CHANGELOG.md` updated under `## [Unreleased]` (user-visible changes only)
