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

### Covered entry points

- **`POST /api/identify`** — a per-face `liveness` verdict; under enforcement
  spoof faces return no `matches` and `blocked=true`.
- **`POST /api/verify`** — the 1:1 verdict carries `liveness`; enforcement
  forces `matched=false` and `blocked=true` for non-real probes.
- **Video / live-stream jobs** — every detection is checked per frame;
  blocked boxes are drawn red, normal boxes green, and each `FaceEntry` carries
  its verdict.

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
