# Contributing

Thanks for considering a contribution to `rs-face`. This document covers the
mechanics of submitting changes; the project's design philosophy lives in
the README under "Limitations / honesty".

## Development setup

```bash
git clone <repo>
cd rs-face
cargo build --release
cargo test --lib
```

The binary needs `ffmpeg` on `PATH` to decode arbitrary video containers.
Image-sequence inputs (`*.png` / `*.pgm` / `*.ppm`) and the synthetic test
source work without any external dependency.

## Project layout

See the README. The module split is intentional: each top-level module
should compile in isolation and have minimal cross-deps beyond `image` and
`integral`.

## Coding conventions

- `cargo fmt` for formatting (CI enforces `--check`).
- `cargo clippy --release --all-targets -- -D warnings` (CI enforces).
- The library is **zero-dep**. Do not add a `Cargo.toml` `[dependencies]`
  entry without prior discussion — the headline feature is the absence of
  one.
- Hand-rolled parsers / decoders are fine and encouraged; pulling in a crate
  for a 50-line codec is not.

## Tests

- Unit tests live next to the code in `#[cfg(test)] mod tests { ... }`.
- Integration tests under `tests/` use the public library API.
- The bench suite (`cargo test --release bench_detect -- --ignored`) is
  expected to take seconds; it is *not* a CI gate but should not regress
  silently.

## Reporting bugs

Please include:

- The exact command you ran.
- The full stderr (with `RUST_BACKTRACE=1` if the panic was unexpected).
- For "no detections" reports: the source frame, the cascade (built-in
  demo / OpenCV XML / `.rfcf`), and `--min-score`. The known limitation
  in the README is the most common root cause.

## Honest disclosures

If your patch makes the demo cascade *more* permissive without also
tightening NMS / score threshold, expect a review comment. The project's
hallmark is that it works without lying about correctness; please don't
add anything that papers over that.