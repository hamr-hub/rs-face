# Silent face liveness (MiniFASNet)

Silent face-anti-spoofing answers one question that detection + recognition
cannot: **is the detected face a live person, or a printed photo / screen
replay?** It is called *silent* because the user does nothing extra — a single
RGB frame is enough.

## Why

A recognizer ranks identity from appearance. A high-quality printed photo or a
replayed video of an enrolled person carries that appearance, so recognition
alone is bypassable. Liveness gates the identity result on a separate signal
(texture statistics MiniFASNet extracts from the crop), closing the spoof
bypass without changing the recognition pipeline.

## Algorithm

The implementation ports the MiniVision `Silent-Face-Anti-Spoofing` design:

1. Take a detected face box and grow it about its centre — a **2.7×** crop for
   `MiniFASNetV2` and a wider **4.0×** crop for `MiniFASNetV1SE`, clamped to
   the image bounds. Context around the face is part of the signal.
2. Resize the crop to **80×80** with bilinear interpolation.
3. Build a raw `[0,255]` **BGR / NCHW** tensor (no mean/std normalization —
   the exported models were trained on the raw range).
4. Run the graph to get three logits and apply a numerically stable softmax.
   Classes are `[printed photo, real face, screen replay]`; **real is index 1**.
5. Average the two models' probability rows and decide.

Two models at different crop scales are averaged (the upstream-recommended
ensemble); a single-model setup is also supported. The decision accepts a face
when the averaged argmax is the real class **and** the real probability clears
`min_real_score`.

### Where the code lives

- `src/liveness.rs` — runtime-free core: expanded crop, BGR/NCHW build,
  softmax, fusion decision. No ONNX dependency; fully unit tested.
- `src/liveness_detector.rs` — `LivenessDetector`: loads the graphs and runs a
  check per face. Behind the `ort-backend` / `tract-backend` feature.
- `src/models.rs` — two pinned `ModelSpec`s with SHA-256 digests
  (`liveness_minifasnet_v2`, `liveness_minifasnet_v1se`), Apache-2.0.
- `tools/fetch_models.sh` — downloads and verifies both graphs.

## Build

Liveness is opt-in; the default build stays zero-dependency. Pick a backend:

```bash
# pure-Rust ONNX runtime (CPU, no C++ toolchain)
cargo test --features tract-backend --test liveness_e2e -- --nocapture

# ONNX Runtime (fastest, GPU-capable)
cargo run --release --features ort-backend -- ...
```

## Platform integration

The platform keeps a zero-dependency default and adds a `liveness` feature:

```toml
# platform/Cargo.toml
liveness = ["rsface/tract-backend"]
```

Build the server with the feature and fetch the models first:

```bash
tools/fetch_models.sh
cargo build --release --manifest-path platform/Cargo.toml --features liveness
```

Configuration (environment):

| variable | default | meaning |
|---|---|---|
| `RSFACE_LIVENESS_ENABLED` | `false` | turn liveness on at runtime |
| `RSFACE_LIVENESS_MODELS_DIR` | `models` | directory with the two ONNX graphs |
| `RSFACE_LIVENESS_MIN_REAL_SCORE` | `0.0` | min real probability to accept (raise toward `0.5` for a stricter gate) |
| `RSFACE_LIVENESS_ENFORCE` | `false` | non-real faces get no identity matches |
| `RSFACE_LIVENESS_QUALITY_GATE` | `false` | reject tiny/blurry/badly exposed/clipped crops before the classifier runs |
| `RSFACE_LIVENESS_TEMPORAL_FRAMES` | `1` | consecutive real frames required in video/stream jobs (`>1` adds temporal defence) |
| `RSFACE_LIVENESS_TEMPORAL_MIN_SCORE` | unset | minimum mean real score over the temporal window; unset uses `RSFACE_LIVENESS_MIN_REAL_SCORE` |

## Quality gate

MiniFASNet relies on high-frequency detail in a reasonably sized, in-focus
crop. A tiny, heavily blurred, badly exposed or contrast-clipped crop has
already lost the cues that separate live from spoof — yet the network still
emits three confident logits from the wrong distribution. The optional
quality gate measures the crop on the **native resolution** (before the
`80×80` resize) and rejects it fail-closed:

- **sharpness** — variance of the 4-neighbour Laplacian (the same focus
  statistic as OpenCV's `Laplacian(...).var()`);
- **resolution** — minimum crop edge, so distant small faces are not scored
  from an up-sample;
- **brightness** — mean luma bounded to reject dark/washed-out frames;
- **clipping** — fraction of pixels pinned to `0`/`255`, which spikes on
  clipped phone replays and contrast-crushed prints.
- **high-frequency energy** (`high_freq_ratio`) — mean-squared residual
  against a 3×3 box blur, normalised by luma variance. Smooth skin is low
  while a screen's pixel grid and moiré patterns raise it, so it is a useful
  replay signal. It is reported **informational only** and deliberately does
  not block: an uncalibrated threshold would false-reject genuinely detailed
  faces. Collect it per crop to calibrate a threshold (the v0.4 data loop)
  before gating on it.

The logic lives in the runtime-free `src/quality.rs` and ships in the default
zero-dependency build; thresholds come from the `QualityConfig::strict()`
preset. Enable with `RSFACE_LIVENESS_QUALITY_GATE=true`. A rejected crop is
reported with label `low quality` and is treated exactly like a spoof verdict
under enforcement.

The statistics are measured on **every** check (whether or not gating is on)
and ride along on the verdict: in the platform JSON each face's `liveness`
object carries a `quality` snapshot (`sharpness`, `mean_brightness`,
`clipped_ratio`, `high_freq_ratio`, crop size), so the replay signals can be
collected alongside real verdicts to calibrate thresholds.

These per-face measurements are also persisted by migration
`0007_face_quality.sql`: the four signals are written to nullable columns on
the `faces` table (NULL when liveness did not run for that face), so
threshold calibration can aggregate data across jobs and server restarts
rather than only observing live API responses.

Migration `0008_face_verdict.sql` additionally stores the face's verdict
(`is_real`, nullable, plus `blocked`) so each stored signal row is labelled by
what the classifier decided. The read-only endpoint
`GET /api/liveness/quality-summary` aggregates the stored rows grouped by
verdict — sample count and mean of each signal for `real` vs `spoof` — giving
the distributions from which to read a separating threshold, across jobs and
restarts. It returns an empty `groups` list when there is no database pool or
no stored verdicts.

## Temporal voting

For video / live-stream jobs, a face can additionally be required to look real
on N **consecutive** frames. Faces are associated frame to frame with the
core `FaceTracker`; each track keeps a sliding `TemporalVote` window, and a
single non-real frame resets the streak. The gate is fail-closed — during the
warm-up the track is not confirmed, so a short clip carrying only one good
frame can never pass enforcement. Set `RSFACE_LIVENESS_TEMPORAL_FRAMES=3` (the
upstream live-stream recommendation) to enable; `1` keeps the plain per-frame
behaviour. The averaged real score over the streak also gives a steadier
confidence than any single frame and must clear `RSFACE_LIVENESS_MIN_REAL_SCORE`,
or the separate `RSFACE_LIVENESS_TEMPORAL_MIN_SCORE` when configured, so one
momentarily high-scoring frame cannot carry otherwise weak frames through
enforcement. The default threshold remains `0.0`, preserving the upstream
argmax behaviour.

### Covered entry points

- **`POST /api/identify`** — a per-face `liveness` verdict; under enforcement
  spoof faces return no `matches` and `blocked=true`.
- **`POST /api/verify`** — the 1:1 verdict carries `liveness`; enforcement
  forces `matched=false` and `blocked=true` for non-real probes.
- **Video / live-stream jobs** — every detection is checked per frame;
  blocked boxes are drawn red, normal boxes green, and each `FaceEntry` carries
  its verdict.

### Fail-closed on error

Under enforcement (`RSFACE_LIVENESS_ENFORCE=true`) the gate needs a positive
real verdict; a **missing** verdict is treated as a spoof verdict. When the
liveness backend is loaded but a check returns nothing — no face box found,
an inference error, or a poisoned shared lock — both `/api/verify` and the
video/stream jobs block the detection rather than letting it through. This
prevents an attacker from triggering an error to slip past the gate. When
enforcement is off the verdict stays purely informational, and with no
backend loaded nothing is ever blocked.

### Metrics

Per-job `JobStats` records `spoof_detections` and `blocked_detections`; the
global metrics endpoint aggregates them across jobs and exports:

- `rsface_spoof_detections_total`
- `rsface_blocked_detections_total`

## Tuning the FAR / FRR trade-off

`min_real_score = 0.0` reproduces the pure-argmax rule (real only has to beat
the two spoof classes). Raise it until the false-reject rate matches your SLA —
a stricter gate lowers the false-accept rate at the cost of rejecting some real
users. Quantitative tuning needs a labelled anti-spoofing benchmark (e.g.
OULU-NPU / CASIA-FASD); the in-tree end-to-end test only sanity-checks a real
frame versus synthesized blur/print/screen variants.
