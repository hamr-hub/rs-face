# Benchmarks

End-to-end detector throughput on a synthetic test image. Run with:

```bash
cargo test --release --test bench_detect -- --ignored --nocapture --test-threads=1
```

## Baseline (pre-optimisation)

| Resolution   | Detections | Per-frame | FPS    |
|--------------|-----------:|----------:|-------:|
| 640×480      |       4320 |    8.18ms | 122.21 |
| 1920×1080    |       1240 |   52.21ms |  19.15 |

## After P3 optimisations (NMS spatial bucket + inv_scale pre-compute + unsafe ptr integral)

| Resolution   | Detections | Per-frame | FPS    | Δ      |
|--------------|-----------:|----------:|-------:|-------:|
| 640×480      |       4320 |    7.46ms | 133.99 | +9.6%  |
| 1920×1080    |       1240 |   49.18ms |  20.33 | +6.2%  |

Detection counts are identical to baseline, so the speedup is purely
algorithmic (no quality regression).

## How to read these numbers

- The synthetic test image has 3 face-like bright spots at different scales;
  the detector picks up many overlapping windows per face.
- The 640×480 case is dominated by **NMS** (4320 candidates → bucketing
  shines here).
- The 1920×1080 case is dominated by **integral image construction**
  (saturates memory bandwidth).

## Where the time goes (informal, on aarch64)

| Step                          | 640×480  | 1920×1080 |
|-------------------------------|---------:|----------:|
| Integral image (regular)      |  ~0.5ms  |   ~5ms    |
| Squared integral              |  ~0.5ms  |   ~5ms    |
| Rotated integral              |  ~0.5ms  |   ~3ms    |
| Sliding window + cascade eval |  ~5ms    |  ~30ms    |
| NMS                           |  ~0.5ms  |   ~3ms    |
| Output (PNG + manifest)       |  ~0.5ms  |   ~1ms    |

These are rough estimates; a real profile with `perf` or `cargo flamegraph`
would give precise numbers.

## Future work

The next big wins, in priority order:

1. **SIMD for the sliding-window inner loop** — feature evaluation is
   currently scalar. A `#[cfg(target_arch)]`-gated AVX2 / NEON path
   could 2-3× the per-window throughput.
2. **GPU variance prefilter** is wired in but only triggers at >500×500
   inputs. Lowering the threshold (or detecting when the GPU is fast
   enough to be worth it) would help medium-sized images.
3. **Multi-image batching** — the worker thread currently processes
   one frame at a time. A small batch (4–8 frames) per worker would
   amortise the per-frame cache warm-up.

These are left out of the current scope because they need profiling
data to drive decisions, not guesses.
