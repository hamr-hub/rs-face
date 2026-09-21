# Changelog

All notable changes to `rs-face` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — multi-recogniser identity consensus
- **`POST /api/jobs/{id}/recognize` exposes weighted-vote face identity
  in the web UI.** The endpoint takes an uploaded image job, runs the
  configured detector, then identifies every detected face with LBPH,
  eigenfaces and Fisherfaces, fusing the per-recogniser outcomes through
  `ensemble::fuse_recognitions`. The response carries each face's
  consensus label (votes, confidence, agreeing sources) plus full
  per-recogniser vote detail (match / below-threshold / ambiguous and
  distance). Results are cached per job, and the gallery is loaded once
  and cached process-wide.
- **`recognize.js` consensus panel**: a topbar toggle (☺) renders the
  consensus over the preview image (labelled boxes with vote count and
  confidence) and lists each face with its recogniser breakdown. No new
  JavaScript dependencies; the choice persists in `localStorage`.
- **Registered-face gallery convention**: identity is supplied through
  `RSFACE_GALLERY_DIR`, a folder of identity folders (one subdirectory
  per person containing `.pgm`/`.ppm`/`.png` crops). docker compose
  mounts `../data/gallery` into the server; a missing or empty gallery
  makes `/recognize` return `no_gallery` without affecting detection
  features. Documented in `platform/.env.example`.

### Changed — docs honesty & rustdoc gate
- **`cargo doc` is now warning-free for both crates and enforced in CI**:
  crate-level docs linked feature-gated ONNX modules (`scrfd`, `arcface`,
  `onnx`, `align`, `models`) with intra-doc links that break on the default
  build — they are now plain code references; the stale “`std` feature” claim
  in `src/lib.rs` was corrected (the crate is `std`-only; threading lives
  behind the `pipeline`/`source` cargo features). Two private-item links in
  `haar/cascade.rs` and four Chinese-doc angle-bracket cases in the platform
  crate (`jobs.rs`, `cache.rs`, `metrics.rs`) were fixed the same way. CI and
  `tools/pre-push.sh` now run
  `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items`
  for both the core and platform crates.
- **Stale GPU/vendor docs refreshed**: `docs/CPU_VS_GPU_REPORT.md` no longer
  lists the deleted `rocm.rs` / `ascend.rs` / `mlu.rs` stubs; CUDA is
  documented as the real `cuda-backend` implementation, and DirectML points
  at the `ort-directml` execution provider. `docs/INDEX.md` says “three
  zero-dep detectors” instead of the long-gone “six detectors”.

### Fixed — correctness audit (Haar / video tracking / PNG)
- **Pyramid `min_size` no longer returns zero detections when the minimum is
  above the 24 px base window.** Footprints grow as the image shrinks, so
  windows larger than the base exist only on later pyramid levels; the scan
  now *climbs* past sub-min levels instead of terminating the pyramid.
- **Tilted (45°) Haar features are now fully supported end to end.** The
  `.rfcf` writer emits version 3 with a per-feature flags byte (bit 0 =
  tilted; v2 files still load as all-upright), `tools/convert_opencv_xml.py`
  carries each OpenCV feature's `<tilted>` flag through, and both eval paths
  score tilted rects through the rotated integral table (`CV_TILTED_OFS`
  corners, zero-border clipping on image rims). Cascades containing tilted
  features force CPU scanning — the GPU kernels have no rotated-table path,
  so previously such features were silently scored as upright.
- **Rotated-integral border queries at image rims now read zero** (row/column
  0 are padding), matching OpenCV's convention; the table is pinned by
  brute-force cone-enumeration and OpenCV-recurrence references.
- **`video_id` embeddings are now sparse index-tagged pairs
  `(detection_index, embedding)`** so a dropped embedding (a face below the
  minimum pixel size) can no longer silently attach later embeddings to the
  wrong tracked detections.
- **PNG decoding rejects unsupported inputs explicitly**: Adam7-interlaced
  images, zero-width/zero-height images, and bit depths other than 8 now
  return descriptive errors instead of decoded garbage.


### Added — ArcFace angular-margin trainer for EmbedNet
- **`src/embednet.rs`** — new `ArcFaceTrainer` that trains the existing
  zero-dependency EmbedNet backbone with the ArcFace additive angular
  margin softmax loss (Deng–Guo–Xue–Zafeiriou, CVPR 2019): per-identity
  L2-normalised classifier rows, target logit `s·cos(θ + m)` with
  `s = 32` / `m = 0.5` defaults, the canonical `cos(π − m)` easy/
  hard-sample gate, and a training-only `classify()` cosine probe. Unlike
  `ContrastiveTrainer`, it trains on ordinary labelled samples rather
  than balanced same/different pairs, giving stronger inter-class angular
  separation on the same 128-d embedding space. Zero deps; the head is
  training-only and inference is unchanged. Two tests prove correctness:
  analytic gradients against central differences (conv4 weight, fc bias,
  target and non-target classifier rows) and a 5-identity toy
  classification convergence run (≥ 90 % held-out accuracy).
### Added — silent face liveness (MiniFASNet)
- **`src/liveness.rs`** — runtime-free core of the MiniVision
  Silent-Face-Anti-Spoofing integration (Apache-2.0): boundary-clamped
  expanded crop (2.7x / 4.0x scales), raw `[0,255]` BGR/NCHW build,
  numerically stable 3-class softmax, and the two-model averaging decision
  with a configurable `min_real_score` gate. Classes are
  `[printed photo, real face, screen replay]`.
- **`src/liveness_detector.rs`** — `LivenessDetector` that loads the
  MiniFASNetV2 (2.7x) + MiniFASNetV1SE (4.0x) ONNX graphs and runs a
  real/spoof check per detected face; single-model loading is supported.
- **`src/models.rs`** — two new pinned, commercially usable
  `LIVENESS_MINIFASNET_V2` / `V1SE` specs (`ModelKind::Liveness`).
- **`src/image/mod.rs`** — `RgbImage::crop` and
  `RgbImage::resize_bilinear`.
- **`tools/fetch_models.sh`** — fetches and verifies the two Apache-2.0
  liveness models.
- Tests: 10 unit tests plus a real-weight end-to-end check
  (`tests/liveness_e2e.rs`) which classifies the Lena face as real with
  `real_score = 0.997`.

### Added — multi-algorithm ensemble fuser
- **`src/ensemble.rs`** — `TaggedDetection` + `fuse()` greedy cluster
  over per-algorithm detection sets. Groups boxes whose pairwise IoU
  exceeds `iou_threshold`, weighted-averages the bbox by
  `source_weight * score`, and emits a cluster iff `votes >= min_votes`.
  Zero deps; reachable from `cargo build --no-default-features --lib`
  because `Detection`/`iou` are core types. The platform's
  `/api/jobs/{id}/compare` endpoint can drive the `sources` field
  directly to render the consensus breakdown.
- **`tests/golden_eval.rs`** — integration test that runs Haar,
  Luminance, CNN (raw + calibrated), and 4 ensemble variants over a
  4-image face-positive set (lena, two-people, demo_face_256, an
  in-memory synthetic face where Haar is known to fail) and prints a
  precision / recall / F1 / avg_ms table. The Haar-gated ensemble is
  the deployment-safe default: it preserves Haar's measured F1
  while letting agreeing detectors refine the box. Two more
  ensemble strategies (`union` and `consensus`) are reported for
  comparison. Labels are gitignored under
  `tests/fixtures/golden/labels/` per the accuracy-subagent contract.
- **`tests/inspect_algos.rs`** — `#[ignore]`d debug helper that dumps
  every detector's output on the golden set so a calibration cycle
  can inspect failure modes without modifying the eval table.

### Added — golden-set eval is now `#[ignore]`-gated
- **`tests/golden_eval.rs::golden_eval_table`** now carries
  `#[ignore = "requires gitignored tests/fixtures/golden/labels/*.txt —
  bootstrap via inspect_algos"]`. The default `cargo test --tests` run
  no longer fails on a fresh clone; the eval is opt-in via
  `cargo test --test golden_eval -- --ignored --nocapture --test-threads=1`.
  This unblocks CI matrix runs that previously blocked on the missing
  gitignored labels.

### Performance — 2.31× single-frame detection speedup
End-to-end `Detector::detect` on a 640×480 grayscale frame, default
preset, 30-iter median:

| step | median (ms) | speedup |
|------|------------:|--------:|
| baseline (pre-merge main) | 300 | 1.00× |
| + variance fast path | 268 | 1.12× |
| + narrow `IntegralImage` dispatch (u32 specialisation) | 180 | 1.78× |
| + precomputed `variance_part` (FMA-friendly) | 135 | 2.22× |
| + cast/refactor polish | **130** | **2.31×** |

- **`src/integral.rs`** — `passes_variance_sums_fast` skips the
  pre-filter check on images whose variance is provably in-range; the
  `IntegralTable` match is hoisted; `rect_sum_unchecked_narrow` skips
  the bounds check when the image fits in a `u32` squared integral.
- **`src/detector.rs`** — per-level constants hoisted, `variance_part`
  precomputed once per scale and threaded through `classify`, GPU
  probe logging (`[gpu] OpenCL probe OK/FALLBACK`) so silent GPU
  failures become diagnosable from stderr.
- **`src/haar/cascade.rs`** — `classify_inbounds_with_variance_part`
  shares one inner loop across narrow / wide integral paths;
  `EvalCache` carries a `narrow_integral` flag.
- **`src/haar/feature.rs`** — kind discriminators (`is_custom` /
  `is_tilted` / `is_wide`) hoisted out of the per-rect loop, the
  redundant `u64 → i64` cast dropped (rect sums are always ≥ 0).
- **`benches/detect_640x480.rs`** — focused single-image criterion
  benchmark for regression tracking.

### Added — `--batch-dir` for photo-folder triage
- **`src/batch.rs`** (new) + `src/main.rs` wires `--batch-dir <DIR>`
  for batch face detection on a directory of standalone images.
  Before, passing a directory treated it as a video sequence with
  monotonic `frame_NNNNN.png` outputs. Now each input image produces
  its own annotated PNG named after the basename
  (`photo1.pgm → detections/photo1.png`) plus a single combined
  `batch_manifest.json` with per-image detection counts, errors, and a
  global stats block (`images_processed`, `images_with_face`,
  `total_detections`, `elapsed_ms`). Per-image decode failures
  (truncated JPEG) are isolated — the manifest records `"error"`
  and the run continues.
- **`tests/batch_dir.rs`** (new, 4 tests) — uses the bundled demo
  portrait to assert `run_batch_dir` produces the expected manifest
  shape, correctly handles missing PNG outputs for face-less images,
  and surfaces per-image errors for truncated input.
- Feature gate: `batch` is on the default feature list (alongside
  `pipeline`); zero-dep build (`cargo build --no-default-features --lib`)
  still passes.

### Added — platform worker health + structured error responses
- **`platform/migrations/0003_jobs_health_columns.sql`** — five new
  columns (`started_at`, `heartbeat_at`, `updated_at`, `archived`,
  `error_code`) with idempotent `IF NOT EXISTS`, plus five indices
  (BRIN on `heartbeat_at` + `updated_at`, B-tree on `status` +
  `(archived, created_ms DESC)` + `error_code`).
- **`platform/server/src/persist.rs`** — `mark_started`, `heartbeat`,
  `set_archived`, `set_error_code`, `reap_orphans(stale_secs)`.
- **`platform/server/src/jobs.rs`** — `run_job` writes `started_at`
  + first `heartbeat_at` via `mark_started`; a tokio task ticks
  every 30 s until `cancel` flips. Cancelled / done / error paths
  stop naturally via the shared `AtomicBool` with the watchdog.
- **`platform/server/src/main.rs`** — after `migrate()` success,
  calls `db.reap_orphans(300)` (10× heartbeat) with a single log
  line; orphaned jobs are marked `error_code='orphaned'`.
- **`platform/server/src/api.rs`** — `error_response_with(code, msg,
  hint)` produces `{"error": "<msg>", "error_code": "<snake>",
  "error_hint": "<opt>"}`. Legacy `{"error": "<msg>"}` shape
  preserved. 7 high-impact codes mapped: `queue_full` (429),
  `upload_too_large` (413), `missing_field` (400), `no_such_job` (404),
  `bad_key` (400), `ssrf_blocked` (403), `media_gone` (410); other
  paths fall back to `error_code="internal"`.

### Added — frontend UX: drag-drop preview + sidebar SSE progress + upload queue
- **`platform/web/upload-queue.js`** (new, 433 lines) — multi-file
  upload queue with `XHR` (real `bytesSent/bytesTotal` progress),
  per-file concurrency (`localStorage.rsface.upload.concurrency`,
  1/2/3/4/6/8), failure-row retry/remove, aggregate progress,
  persistence across reloads. Old `submitImage(file)` API still
  works (now goes through the queue).
- **`platform/web/dropzone-preview.js`** (new, 153 lines) — once a
  user drops / picks files, the dropzone shows count + size + a
  56×56 thumbnail of the first image before submission; ✕ button
  resets the underlying `<input type="file">`. Video dropzone
  shows icon + name + size (no thumbnail decode).
- **`platform/web/index.html` + `app.js`** — each `.sb-item` card
  gains a 2px SSE-driven progress bar; `running` pulses, `done`
  fills 100% in `--success`, `error/cancelled` fills in `--danger`,
  hidden otherwise. Stream tasks where `frame_count` is unknown
  fall back to a 2-95% pulse so the user always sees motion.
- **`platform/web/style.css`** — new `.seq-*` / `.sb-prog-*` /
  `.dz-preview-*` classes using existing CSS variables, so the
  dark/light/auto theme switch keeps working.

### Added — test / benchmark / CI hardening
- **6 new integration test suites** (1655 lines, all default-feature
  green):
  - `tests/nms_edge_cases.rs` (9) — full overlap, threshold 0/1,
    many-vs-one cluster, modern + classical paths
  - `tests/iou_edge_cases.rs` (12) — degenerate w/h, inverted boxes,
    12×12 = 144-pair symmetry matrix, NaN through classical NMS
  - `tests/integral_image_tests.rs` (12, `detector-haar`) — oracle
    vs brute-force reference, wide `u64` path coverage (4500×4500)
  - `tests/resize_tests.rs` (14) — bilinear / area / downscale at
    1×1, 2×2, odd, 1:100, 4×4 ramp→2×2 hand-checked
  - `tests/algo_compat.rs` (7, `detector-haar + detector-luminance`,
    CNN opt-in via `RSFACE_RUN_CNN_COMPAT=1`) — schema invariants
    across algorithms, zero-size edges, brand strings
  - `tests/cnn_scratch_tests.rs` (11, `detector-cnn`) — `CnnScratch`
    size contract, buffer reuse, standalone `conv2d_into` /
    `maxpool2_into` / `fc_into` / `relu` / `sigmoid` sanity
- **3 new criterion benches** (`harness = false`):
  - `benches/detect_640x480.rs` — Haar on synthetic 640×480, 3 soft
    blobs
  - `benches/iou_bench.rs` — 1000-call IoU batches (integer +
    sub-pixel)
  - `benches/integral_bench.rs` — `from_gray` + 1000-query batches
    at 640×480 and 1920×1080
- **`.github/workflows/ci.yml`** — new `benches` job runs
  `cargo bench --no-run` + `cargo test --no-run` across 4 feature
  combos (default / `detector-haar` / `detector-luminance` /
  `detector-cnn`) so `[[bench]]` / `[[test]]` required-features
  drift is caught before PR time. Existing `core` job gained a
  `cargo test --tests` step so every `tests/*.rs` target gates the
  merge.
- Test counts: `cargo test --lib` **201** (+15), `cargo test --tests`
  **282** (+218 across 23 suites), platform **54** (unchanged).

### Changed — code-quality sweep (no behaviour change)
- **`src/main.rs`** — `run_cnn_pipeline` and `run_algo_pipeline` were
  near-identical (~120 lines of duplicated frame-loop / RGB-fallback /
  record-building / manifest-write code). Replaced by a single shared
  `run_with_detector` driver; the CNN path now adapts its `&[f32]` /
  `CnnDetection` contract to the shared `Fn(&GrayImage) -> Vec<Detection>`
  signature via a small closure. Same manifest layout, same `--only-with-face`
  semantics, same `--out` paths.

### Fixed — error chain consistency
- **`OnnxError::source()`** (`src/onnx/mod.rs`) now exposes the wrapped
  `io::Error` as the error source. Previously only `Display` carried the
  underlying message; `Error::source()` returned `None`, breaking any
  generic error chainer that walks `source()`.

### Added — edge-case tests
- **`src/face.rs`** — `iou_nested_box` (a box fully contained in another
  should give `area(inner)/area(outer)`) and `iou_is_symmetric`
  (`a.iou(b) == b.iou(a)` across disjoint, partial-overlap and nested
  cases). The pre-existing `iou_*` tests covered identical, disjoint
  and half-overlap but not these.
- **`src/detector.rs`** — `detector_with_zero_area_image_returns_no_detections`
  covers `0×0`, `10×0`, `0×10` zero-area images so future pyramid /
  scan refactors cannot regress the empty-source path.

### Security — platform server hardening
- **Uploads now stream to a staging file instead of buffering in memory**
  (`platform/server/src/api.rs`, `stream_field_to_staging`): the multipart
  handler read the whole body via `field.bytes()` and then cloned it into a
  `spawn_blocking`, so a few concurrent multi-GB uploads OOM-killed the
  server. The field is now chunk-streamed to `tmp/staging/` with a running
  size counter that rejects past the configured limit, the job is created
  (queue slot counted) only after the full file lands, every prep-failure
  path rolls index/DB/workdir/staging back, and startup sweeps the staging
  directory of leftovers from crashed clients.
- **`file://` removed from the accepted stream URL schemes**: an
  unauthenticated client could point the ffmpeg-backed stream job at any
  local file and watch it through SSE. RTSP/HTTP(S) LAN-camera use cases
  stay; non-`test://` URLs must now parse to a non-empty host
  (`url_authority_host`, userinfo/IPv6/port-aware) and are length-capped.
- **Media endpoint path traversal closed**: `local://` keys are rejected
  when absolute or when the canonicalised path escapes the media root, so
  `local:///etc/passwd` and `local://../` style keys no longer read
  arbitrary files.
- **Telemetry is PII-filtered server-side** (`is_safe_event`): event names
  and props are whitelisted/sanitised before they reach the JSONB column, so
  tokens, email-like values and free-form input cannot be exfiltrated
  through the analytics endpoint.
- **SSE connection cap**: `/api/jobs/{id}/events` had no bound on concurrent
  connections — each one spawned a task, opened a broadcast subscription
  and could pin memory. A process-global counter now limits connections to
  128 with HTTP 429 past the cap; terminal-event detection is typed
  (`type` ∈ done/error/cancelled) instead of substring matching, and the
  duplicate axum/SSE-layer keepalives are reduced to one.
- **All error responses use one JSON envelope** (`{"error": ...}`) via
  `error_response`, replacing the mix of plain-text bodies across job
  detail/cancel/media/import.

### Added — platform
- **`download.zip` export**: a dependency-free STORE-method ZIP writer
  (`platform/server/src/zip.rs`, local headers + central directory + EOCD,
  const-built CRC-32 table) streams a job's face crops as one archive; unit
  tests pin the record layout and known CRC vectors.
- **"Compare (TB)" trigger on the job page** (`platform/web/`): a dedicated
  button fires `tb-compare` into the existing compare pipeline, with
  governed polling (tab-visibility aware) instead of an unconditional timer.

### Fixed — platform runtime
- **Job deletion now reclaims media**: deleting a job (single or batch)
  fire-and-forget lists the `jobs/{id}/` S3 prefix and deletes every object,
  then removes the local media and tmp work directories. The new
  `list_objects` (continuation-token aware) / `delete_object` methods also
  fixed a second SigV4 defect — canonical query parameters were sorted and
  percent-encoded for signing but sent raw, which produced
  `SignatureDoesNotMatch` whenever a key (e.g. continuation token) needed
  encoding.
- **O(n²) frame scan removed from the video loop**: per stored crop the job
  summed face counts across all frames under a lock; replaced with a single
  running counter.
- **Migration runner hardened** (`platform/server/src/persist.rs`): a new
  `schema_migrations` table records each applied file, files run inside one
  transaction (a failing statement rolls the whole file back instead of
  leaving a half schema plus only a log line), and statement splitting is
  lexer-aware — nested block comments, line comments, quoted strings and
  `$tag$` dollar-quoted bodies no longer split on inner semicolons.
  Migration failure is now fatal at startup rather than a swallowed warning.
- **Poison tolerance across aggregate reads**: if a worker panics while
  holding a per-job lock, the job list, metrics and aggregate-sample paths
  recover the last-known value instead of panicking and wedging the endpoint
  for every subsequent request.

### Changed — platform deployment
- **Multi-arch image builds fixed** (`platform/Dockerfile`): the build target
  was hardcoded to the arm64 musl triple; `TARGETARCH` now maps to
  `aarch64`/`x86_64-unknown-linux-musl` (unknown arches fail explicitly).
- **`docker-compose.yml`**: rustfs/postgres are pinned to
  `rustfs/rustfs:1.0.0` and `postgres:16.15-alpine`, their ports bind on
  `127.0.0.1` only (consoles no longer exposed to the LAN), the Postgres
  password is parameterised via `.env` (`POSTGRES_PASSWORD`,
  alphanumeric-only) with `DATABASE_URL` referencing the same value, and
  `.env.example` documents every knob including timeouts and memory limits.
- `.dockerignore` rewritten to keep the build context small (target/data/
  models/web-dist excluded).

### Fixed — deploy on a fresh checkout works end to end
- **`platform/Dockerfile` cascade path synced with the weights move**: PR #3
  shipped the cascade inside `src/weights/`, but the Dockerfile still
  `COPY`ed `cascade.rfcf` from the repo root. Any fresh checkout failed at
  the asset step; only because CI builds the crate (never the image) did
  this land on main. Path now points at
  `src/weights/haarcascade_frontalface_default.rfcf`.
- **`Dockerfile` cache layer stubs cargo target placeholders**: PR #3 added
  explicit `[[bin]]/[[bench]]/[[example]]/[[test]]` targets (per-algorithm
  `required-features` trimming). Cargo validates target paths at manifest
  parse even for path deps, and the cache-prefetch layer only stubbed
  `lib.rs` + `main.rs`, so it failed with "can't find `cascade_dump`
  example" before any compile happened. The cache layer now creates empty
  placeholders from the manifest itself; the real layer `COPY`s
  `examples/`, `benches/`, `tests/`.
- **`platform/Dockerfile.gpu` mirrors the same asset/target fix** so the GPU
  build stays in lock-step with the standard image.
- `README.md`, `benches/perf_compare.rs`, `examples/{cascade_dump,
  lena_classify_stages}.rs`, `tests/e2e_stress.sh`, `platform/DOCKER_SIZING.md`,
  `platform/MINIMUM_CONFIG.md`, `platform/docs/PLATFORM_DESIGN.md` had stale
  `cascade.rfcf` paths — all refreshed.
- `platform/.env.example` documents the new `SERVER_STOP_GRACE` knob.
### Added — every algorithm is now a cargo trimmable
- **Per-algorithm feature flags** turn the monolithic build into a pick-and-mix
  crate while `default` keeps the exact 0.2.x surface:
  - detectors: `detector-haar`, `detector-luminance`, `detector-cnn`
  - recognisers: `recognizer-lbph`, `recognizer-eigenface`, `recognizer-fisherface`
  - sources: `source` (trait + image sequence + synthetic), `source-http`, `source-ffmpeg`
  - orchestration: `output`, `pipeline`, `video-id`

  Example: `--no-default-features --features recognizer-lbph,source` builds a
  tiny LBPH-only library. Every bin / example / integration test declares
  `required-features`, so trimmed builds skip targets they cannot compile;
  CI gains a 13-combination feature matrix that runs each module's own tests
  on top of the minimal core (now ~45 tests for core-only; 297 with
  `tract-backend`). The unused detection-vector cache was removed from
  `pool`.

### Changed — module groundwork for per-algorithm features
- `Detection` / `non_max_suppression` / `iou` now live in the detector-agnostic
  `rsface::face` module; `rsface::detector` re-exports them, so existing
  `rsface::detector::Detection` imports keep working. The `FaceRecognizer` /
  `IncrementalRecognizer` impls moved out of the trait module into the
  `lbph` / `eigenface` / `fisherface` algorithm modules they belong to.
  Behavior and public paths are unchanged; this only unblocks compiling
  individual algorithms behind cargo features.

### Changed — package hygiene
- The crates.io tarball no longer ships the multi-MB photo fixtures only the
  opt-in ONNX e2e test needs (`biden.ppm`, `two-people.ppm`) nor the rendered
  `docs/samples/` PNGs: package payload drops from ~14.3 MB to ~2.5 MB. Repo
  checkouts and CI are unaffected; `lena.ppm` + the `include_bytes!`-embedded
  `demo_face_256.pgm` used by the default test suite still ship.
- All intra-doc links now resolve under `cargo doc`; the pasted crate-level
  clippy-allow header was removed from the `cnn_train` bin (the workspace
  `[lints]` table already applies to every target).
### Added — zero-dependency deep embeddings (`embednet`)
- **`rsface::embednet`**: a genuinely trainable embedding CNN in pure
  `std` — im2col/col2im GEMM convolutions (1→24→48→64→96 channels),
  max-pool, a 128-d FC head with L2 normalisation, He init, and a full
  analytic backward pass (gradients verified against central finite
  differences). The `ContrastiveTrainer` optimises the Hadsell–Chopra–
  LeCun contrastive pair loss with Adam; `EmbedNetRecognizer` adapts the
  resulting unit vectors to the same zero-dep `embedding::Gallery`
  matcher as ONNX ArcFace and to the grey-crop `FaceRecognizer` trait.
  Weights persist to a strict-versioned `.rsen` container.
- **`embednet_train` binary**: trains on a folder-of-folders
  (`root/<label>/*.{pgm,ppm,png}`), reports held-out pair accuracy and
  same/different distances every 200 steps, and saves the best snapshot.
- **`examples/recognise_embednet.rs`**: end-to-end demo — trains on
  synthetic identities, identifies a held-out probe through the uniform
  trait, and round-trips weights through `.rsen`.
- Honesty scope, same as the CNN detector: trainable from scratch with no
  bundled weights; ONNX ArcFace stays the industrial-accuracy option.

### Added — real OpenCV cascade bundled; zero-arg install works
- **The classical OpenCV frontal-face cascade now ships inside the binary**
  (`src/weights/haarcascade_frontalface_default.rfcf`, converted from OpenCV
  4.10.0's Apache-2.0 XML; source hash pinned in `src/weights/NOTICE.md`).
  Embedded via `include_bytes!` and served by
  `rsface::haar::bundled::bundled_frontalface_cascade()`, the CLI uses it by
  default — no `--cascade` path and no model download are needed to detect
  real faces. The 930 KB audit XML is kept beside it but excluded from the
  crates.io package.
- **`rs-face demo`** is a zero-argument post-install smoke test: the bundled
  cascade runs against an embedded 256×256 portrait (fixture under
  `tests/fixtures/`), prints the detection, writes an annotated PNG to
  `rsface-demo/`, and exits non-zero when no face is found.
- **Uniform `FaceRecognizer` trait** (`src/recognizer.rs`): LBPH /
  eigenfaces / Fisherfaces now dispatch behind one API with a shared
  `Recognition` outcome (including incremental enroll); cookbook snippet in
  `examples/recognise_uniform.rs`.

### Removed — placeholder-weight detectors
- **`src/mtcnn.rs`, `src/hog_face.rs`, `src/yunet.rs` and the random-weight
  blobs under `src/weights/` were deleted.** They shipped 1–3 KB of random
  bytes as "trained" weights and returned zero detections on every frame;
  apologetic labels (`Maturity::Scaffold`, `[SCAFFOLD — detects nothing]`)
  cannot make a non-algorithm useful. The zero-dep detector set is now
  honestly three: `haar`, `cnn` (a genuinely trainable starter net —
  `cargo run --bin cnn_train`), and `luminance`. The `Maturity::Scaffold`
  enum variant and its labelling machinery were removed with them.
- The real, Apache-2.0 **YuNet 2023mar ONNX model stays pinned in
  `src/models.rs`** (SHA-256, fetchable via `tools/fetch_models.sh`) for a
  future ONNX runner; only the fake in-tree implementation is gone.
- `benches/perf_compare.rs` no longer prints YuNet/MTCNN/HoG N/A rows for
  algorithms that are not in the crate.
- Plain `cargo run` now launches the CLI (`default-run = "rs-face"`).

### Changed — reference parity
- **SCRFD preprocessing matches InsightFace's `scrfd.py` exactly**: Python-int
  letterbox with the long side pinned to the input size (and the reference
  quirk of applying the y-derived scale to both axes), and cv2
  INTER_LINEAR-equivalent resampling (centre mapping, 4-tap bilinear, edge
  replication) instead of nearest neighbour. Regression tests pin both.
- `Cascade::eval_stage` now reports the same verdict as the production
  classifier: inner-`normrect` variance normalisation, the
  `value < threshold ? left : right` leaf rule (the legacy `sign` field is
  stored for format round-tripping but never consulted), and the effective
  threshold `stage_threshold + stage_bias`.

### Changed — LBPH now matches OpenCV's `elbp_` bit-for-bit (gallery format v2)
- **The LBP sampler was rewritten to the exact OpenCV `face::LBPHFaceRecognizer`
  convention**, verified against `lbph_faces.cpp` in OpenCV 4.x, 3.4.20 and the
  2.4.9 `facerec.cpp`: neighbour `n` sits at angle `2πn/P` with bit 0 at the
  **3 o'clock** sample (the previous build used 6 o'clock), neighbours are
  **bilinear-interpolated at every radius** with OpenCV's floor/ceil corners and
  `w1..w4` weights (negative row offsets floor like the C++ `cvFloor(int)`
  cast), the comparison is `v > center || |v−center| < f32::EPSILON`, and
  spatial cells are fixed `interior/grid` rectangles whose right/bottom
  remainder strip is dropped. No edge clamping exists upstream and none here:
  only interior pixels (`radius..size−radius`) contribute. Two new unit tests
  pin the ring geometry and the 3 o'clock bit-0 code.
- **χ² distance now returns `None` only when *both* descriptors have zero
  mass** in a cell (the crate's own border-degenerate all-zero descriptor); a
  one-sided zero mass yields the finite sum, matching OpenCV behaviour.
- **The on-disk gallery format is now `RSLB` v2; v1 is rejected** with
  `LbphStoreError::UnsupportedVersion(1)`. v1 was only ever written by
  pre-release builds and its uniform-bin permutation under the old sampling
  convention is not distance-comparable, so mixing the two silently would
  corrupt rankings. The header layout itself is unchanged.
- **Benchmarks are now deterministic**: `bench_lbph` walks crop directories in
  sorted path order — `fs::read_dir` order varied between runs, and exact χ²
  ties (integer-count histograms) made strict LOO rank-1 fluctuate by up to
  four probes. Canonical output is now byte-identical across consecutive runs.
- **All accuracy numbers were re-measured honestly under the new sampler**
  (hard gallery, 77 crops / 21 identities / 68 repeated probes): LOO rank-1 is
  **60/68 = 88.2 %** (was 62/68 = 91.2 % under the old sampler — the old and
  new numbers are not comparable); the easy no-regression gallery is 32/33 at
  the shipped 6×6 grid (finer grids reach 33/33). At `DEFAULT_MAX_DISTANCE`
  16.7: FAR 0.00 %, FRR 50.8 %, pair accuracy 96.6 %; closest impostor 17.5;
  EER ≈ 23.2 (FAR ≈ FRR ≈ 19.7 %). The 9-point sweep now shows 6×6, 8×8 and
  10×10 grids statistically tied (all up to 60/68); 6×6 stays the shipped
  default for its smaller descriptor, and the tie is documented as such.
  Docs refreshed: `docs/recognition-lbph.md`, `docs/gallery-persistence.md`,
  `docs/algorithms.md`, `README.md`.

### Added — zero-dependency trained-model persistence (eigenfaces / Fisherfaces)
- **Trained subspace recognisers survive restarts without retraining**
  (`src/subspace_store.rs`, new module, `src/binio.rs` shared with the LBPH
  codec): `EigenfaceRecognizer` and `FisherfaceRecognizer` gain `to_bytes` /
  `from_bytes` / `save` / `load`. The blobs carry new magics **`RSEF` v1**
  (mean, unit eigenvectors, per-axis inverse-eigenvalue Mahalanobis scales,
  projected coefficients, metric byte, variance-kept, suggested threshold) and
  **`RSLD` v1** (mean, ≤ C−1 discriminant axes, projected coefficients).
  Everything is stored as raw IEEE-754 `f32` bits, so reload is bit-exact and
  rankings are unchanged (round-trip tests assert it).
- Both decoders are untrusted-input boundaries: magic/version checks, bounds-
  checked reads, finite/non-negative scalars, `variance_kept ∈ (0,1]`, exact
  vector lengths, `coeffs == k` per member, up-front allocation cap
  (`MAX_MODEL_FLOATS = 67 108 864`), count/label sanity caps, and
  no-trailing-bytes; errors come back as `SubspaceStoreError`
  (`std::error::Error`), never panics. Saves use the same sibling
  `.<name>.tmp` + atomic-rename mechanism as the LBPH gallery.
- Layout documented byte-by-byte in `docs/gallery-persistence.md` §7–11
  (common 19-byte header, per-format extensions, member sections; ≈ 0.84 MB for
  a k=50 eigenfaces model on the current 77-crop gallery and ≤ ≈ 345 KiB for
  Fisherfaces); cross-feeding an `RSEF` blob to the Fisherfaces decoder (or any
  other permutation) fails with `BadMagic`.

### Added — project governance (enforced, not just documented)
- **`.github/workflows/ci.yml` strengthened**: the existing CI covered
  build/test/zero-dep but had no `fmt`/`clippy` gate and never built the
  platform crate. Now both crates run fmt + clippy `-D warnings`; the core
  job keeps the zero-dep build/test, full build, and synthetic smoke test;
  a new platform job builds + tests `platform/` (ubuntu + macOS matrix kept).
- **Local gates under `tools/`** installed via `bash tools/install-hooks.sh`:
  `pre-push.sh` runs the same checks as CI on both crates; `commit-msg.sh`
  enforces `<type>(<scope>): <summary>`.
- **`GOVERNANCE.md`** — single source of truth for branches, the gate,
  commit rules, zero-dep/structure hard rules, accuracy-evidence
  requirements, and a scope-of-authority table: **bots/agents push branches
  and open PRs, they do not push to `main`**; only the owner merges.
- PR + issue templates under `.github/`.
- One-shot `style: cargo fmt` baseline commit so the new fmt gate starts green.

### Fixed
- **S3 SigV4 signature mismatch** (`platform/server/src/s3.rs`): the signed
  canonical headers included `host` and every `extra_headers` entry
  (content-type, range), but the actual ureq request did not send an explicit
  `Host` and skipped non-content-type extras — the server then rejected
  requests with `SignatureDoesNotMatch`. The request now sends `Host`
  (non-default port included) and all signed extra headers, so the sent
  headers always match what was signed. Added a `host_of` regression test.
- **SSRF allowlist bypass via full-form IPv6 link-local**
  (`platform/server/src/api.rs`, `is_blocked_host`): the link-local check was
  a `strip_prefix("fe") + len == 1` string hack that only matched the
  compressed spelling `fe8::1` and let `[fe80::1]` / `febf::…` through.
  Replaced with a proper fe80::/10 test on the first hextet's top 10 bits;
  added cases for fe80..febf full forms and negative cases (fc00::/7,
  global IPv6).

### Housekeeping — repo audit + directory standards
- **Removed orphan planning docs**: `TASK_PLAN.md` (root) and
  `core/MULTI_ALGO.md` (along with the empty `core/` directory). Both were
  internal Chinese working documents from earlier fix passes; their content
  is superseded by `docs/algorithms.md`, `docs/architecture.md`, and the
  commit history. Repository convention now explicit: working planning
  docs do not belong at the repo root or under `core/`.
- **Wired up the dormant `platform/web/compare.js`**: the "algorithm
  compare mode" frontend module (289 lines) was complete but never loaded.
  Added `<script defer src="/compare.js">` to `platform/web/index.html`
  (between `visibility.js` and `telemetry.js`); updated the file's header
  comment to reflect the actual loading contract. The matching backend
  endpoint `POST /api/jobs/{id}/compare` is already in production.
- **Moved frontend dev tooling back to the repo root** to fix an
  inconsistency in commit 8add57a (`chore(structure): move vite/pnpm
  tooling to platform/web-dev/`): the commit's commit-message claimed the
  five files moved to `platform/web-dev/`, but the actual `git ls-tree -r`
  shows them at the repo root. The `Makefile web-*` targets introduced in
  8add57a called `cd platform/web-dev && pnpm <cmd>` against a directory
  that did not exist, breaking `pnpm install` / `pnpm dev` /
  `web-build`. The fix: keep the dev files at the root (where they
  always were), drop the `WEB_DEV_DIR` Makefile variable, call `pnpm`
  directly. `vite.config.js` paths updated: `root: '../web'` →
  `root: 'platform/web'`, `outDir: '../../web-dist'` →
  `outDir: 'web-dist'`.
- **`CONTRIBUTING.md § Project layout`**: full directory tree + per-file
  responsibilities + 7 conventions ("top-level dirs are sparse by design",
  "no planning docs at the root", "`src/bin/` is for `[[bin]]` targets",
  "`platform/web/` is zero-build", "`platform/testdata/` is split", etc.).
  Previously the section was a one-liner pointing at the README.
- **`README.md § Project layout`**: rewritten to show the full tree
  (with platform/ sub-tree, tests/, benches/, examples/, docs/, tools/,
  data/) and point at `CONTRIBUTING.md` for the canonical version.
- **No new top-level directories (STRUCTURE §2.1)**: the feature branch that
  added the bundled cascade had introduced `assets/`; its contents moved to
  the documented homes — cascade weights + `NOTICE.md` under `src/weights/`,
  the embedded demo portrait under `tests/fixtures/` — with `include_bytes!`
  paths, the `.gitignore` rfcf exception and the crates.io package excludes
  updated. `docs/CASCADE_FIX.md`, the last `*_fix.md` scratch note banned by
  STRUCTURE §2.2 (its uppercase name slipped past the earlier audit), was
  deleted and its three inbound links cleaned.

### Fixed — platform deploy permissions
- **`platform/docker-compose.yml` — rustfs 容器锁定 UID:GID = 1000:1000**:
  rustfs 镜像默认以 root 运行,绑定 `../data/rustfs` 后容器创建的对象
  在宿主侧是 `root:root`,hyx 用户后续 `rm` / `du` / `rsync` 等操作会
  Permission denied。锁定 `user: "1000:1000"` 后内外一致。迁移期
  已有 root-owned 的 `data/rustfs/` 内容需要 `sudo chown -R 1000:1000`
  才能被新容器读到(详见 compose 注释 + DOCKER.md troubleshooting)。

### Housekeeping — structure audit + frontend package simplification
- **Removed `platform/CHANGELOG_CNN.md` and `platform/CHANGELOG_PERF.md`**
  — STRUCTURE.md §2.3 forbids `CHANGELOG_*.md` under `platform/`. Release
  history lives in the root `CHANGELOG.md`. The `docs/INDEX.md` entry that
  pointed at `platform/CHANGELOG_CNN.md` is removed.
- **Removed `platform/web/CHANGELOG_DOUBAO.md`,
  `CHANGELOG_ENHANCE.md`, `CHANGELOG_FIXES.md`** — same rule applies
  under `platform/web/`.
- **Frontend package management simplified**: dropped
  `pnpm-workspace.yaml` (placeholder content, no actual workspace
  defined) and `.npmrc` (broken `onlyBuiltDependencies[]=` syntax).
  Frontend dev workflow is now `pnpm install && pnpm dev` with a single
  `package.json` declaring one `devDependency` (`vite`).
- **Fixed `vite.config.js` comments** that referenced `make docker-up` /
  `make web-dev` — STRUCTURE.md §2.8 forbids a Makefile; the comments
  now point at the canonical `docker compose` / `pnpm dev` commands.
- **Documented `src/weights/`** in `STRUCTURE.md` §1.1 lookup table as
  the home for bundled zero-dep binary weights (`include_bytes!`-loaded).
- **Cleaned up generated / cached files**: removed `.meta.yaml` and
  `cascade.rfcf` from the working tree (both `.gitignore`d; re-generated
  on demand by their respective tools).

### Documentation — Docker is the canonical deployment story
- **`CLAUDE.md` at the repo root codifies the rule**: platform services
  (rustfs + postgres + rsface-server) are deployed / started / integration-tested
  **only** via Docker; algorithm-core `cargo test` / `clippy` / `bench` keep their
  native-cargo fast-iter path (CI remains cargo, ubuntu + macOS matrix).
- **New authoritative ops guide `platform/DOCKER.md`** covers deploy / start /
  verify / e2e-test / PG backup & restore / data migration from old named
  volumes / troubleshooting / cleanup. `platform/README.md` slimmed down to an
  entry-point that links to it.
- **`README.md` gains a "Run as a service (Docker)" section** in the 5-minute
  walkthrough, plus the canonical `docker compose -f platform/docker-compose.yml up -d --build` one-liner.
- **`CONTRIBUTING.md` gains two new sections**: "Working with the platform
  services (Docker)" (the only allowed way to run `rsface-server`) and
  "Frontend development (pnpm dev + hot reload)" (Vite dev server with proxy
  to the Docker backend).
- **`Makefile` gains 7 docker-* targets and 3 web-* targets** as thin wrappers
  around `docker compose -f platform/docker-compose.yml` and `pnpm`:
  `docker-up / docker-down / docker-ps / docker-logs / docker-test /
  docker-restore-pg / docker-clean` and `web-install / web-dev / web-build`.
  *(Subsequently removed in 2026-09; commands are now documented inline.)*
- **`platform/scripts/docker-smoke.sh`** is the new e2e smoke body —
  `/api/health` → rustfs health → postgres `pg_isready` + jobs count → optional
  image upload.
- **`platform/docker-compose.yml` broken reference fixed**: the inline comment
  pointing at a non-existent `./migrate-pg.sh` now correctly directs users to
  `docker exec -i rsface-postgres pg_restore ...`.
- **`data/` bind mounts confirmed as the canonical data path**:
  `data/{rustfs,pg/pgdata,media}/` next to the repo, visible + rsync-friendly.
  The old docker named volumes `platform_rsface-media / platform_pg-data /
  platform_rustfs-data` (introduced in v0.1) are retired; the migration recipe
  lives in `platform/DOCKER.md`.
### Added — zero-dependency LBPH gallery persistence
- **LBPH galleries now survive restarts without the original crops**
  (`src/lbph_store.rs`, new module): `LbphRecognizer::to_bytes` /
  `from_bytes` / `save` / `load` serialise config + per-identity LBP histogram
  descriptors in a checked little-endian binary format (magic `RSLB`,
  versioned). Descriptors are stored verbatim as IEEE-754 `f32`, so the
  round trip is **bit-exact** — chi-square distances and rankings after
  `load` are identical to before `save` (asserted in tests).
- `save` is crash-safe: encode to a sibling `.<name>.tmp`, then atomic
  rename over the destination; rename failure best-effort removes the temp
  file. `decode` treats the blob as an untrusted boundary and validates
  magic/version, every length (truncation), UTF-8 non-empty NUL-free unique
  labels, config sanity (`radius ≥ 1`, `face_size ≥ 2·radius+1`, grid
  non-empty, finite non-negative thresholds), descriptor shape
  (`cells == grid_x·grid_y`, `len == cells·59`), finite values, sanity caps
  on counts, and the absence of trailing bytes; errors are returned as
  `lbph_store::LbphStoreError` (`std::error::Error`), never panics.
- Format documented byte-by-byte in `docs/gallery-persistence.md` (layout,
  size ≈ 8.5 KB/crop at the default 6×6 grid, validation list, atomicity
  guarantees, forward-compatibility policy); 7 codec unit tests; the
  `recognise_lbph` example now ends with a save/load round-trip asserting
  unchanged rankings.

### Added — zero-dependency recognition (Fisherfaces / LDA)
- **Fisherfaces recogniser in the default build** (`src/fisherface.rs`): the
  Belhumeur–Hespanha–Kriegman (PAMI 1997) class-discriminative counterpart to
  eigenfaces — crops are reduced to the leading `n−C` total-scatter PCA directions
  (Gram trick), then the at-most-`C−1` Fisher axes come from the symmetric whitened
  problem `S_W^{−1/2} S_B S_W^{−1/2}`; near-zero eigenvalues of `S_W` use a
  pseudo-inverse weight (floor `1e-9 · λ_max`) so near-duplicate frames and singleton
  classes are handled without any BLAS. Euclidean nearest neighbour, multi-shot
  best-member scoring, verification, and the same
  `Match`/`BelowThreshold`/`Ambiguous`/`NoCandidates` policy as the other recognisers,
  plus `FisherfaceError` (`TooFewSamples`/`SingleClass`/`DegenerateGallery`) and 4 unit
  tests. No weights, no third-party crate.
- The cyclic-Jacobi symmetric eigensolver is now shared in `src/linalg.rs`
  (`pub(crate)`, same `1e-10` relative tolerance / 30 sweeps), used by both eigenfaces
  and fisherfaces; behaviour-preserving extraction (all eigenface tests unchanged).
- **Measured real-face accuracy** (`docs/recognition-fisherface.md`,
  `docs/bench-results-fisherface.md`): strict per-probe LOO with full PCA+LDA retrain
  gives **59/68 = 86.8 % rank-1** on the 77-crop / 21-identity hard gallery — one
  probe above the sweep-proven 58/68 eigenfaces/PCA ceiling, three below LBPH's
  62/68 — and **33/33** on the easy 35-crop set. The pair-distance **EER ≈ 12.8 % is
  the best of the three zero-dep recognisers** (eigenfaces 14.0 %, LBPH 22.5 %),
  making Fisherfaces the strongest classical verification option; the stored
  descriptor is only `C−1 ≤ 20` `f32`s. The calibrated
  `fisherface::DEFAULT_MAX_DISTANCE = 3.0` is a conservative low-FAR point on the hard
  gallery (FAR 0.40 %, FRR 48.2 %) and a near-perfect point on the easy one
  (FAR 0 %, FRR 2.4 %). Histogram equalisation lowers EER further (≈ 6.9 %) but
  costs two rank-1 probes, so raw stays the default; the 64-px crop size wins a
  32/48/64 sweep outright.
- `bench_fisherface` bin (default features; registered explicitly in `Cargo.toml`):
  strict-LOO raw/equalised canonical rows plus a crop-size sweep, pair distributions,
  EER / best-threshold / crate-default operating points, rank-1; writes the markdown
  report. `tools/lbph_prep.sh` gains it as stage 5.
- `recognise_fisherface` example (`cargo run --example recognise_fisherface`):
  train-once gallery + identify / rank / verify on synthetic crops, no downloads.


### Added — larger real-face evaluation gallery
- **Eval gallery expanded from 35 crops / 8 identities to 77 crops / 21
  identities**: two additional short-drama clips (`dramabox`, `shorttv`) join the
  three originals, 150 sampled frames in total (30 per clip), 77 of them passing the
  SCRFD face gate — 195 same-identity / 2 731 different-identity pairs, with harder
  pose and lighting than the original set. Clusters whose ArcFace join cosine sat in
  the 0.48–0.55 grey zone were audited visually with per-identity contact sheets; all
  are same-actor variation, no label mismatch accepted.
- The original 35-crop gallery (`goodshort` + `reelshort` + `vibeshort`) is kept in
  the lab tree as `out/lbph/crops_old`, an **easy no-regression set** every tuning
  change must not regress.
- **Hyperparameter sweeps in the accuracy benches**: `bench_lbph` now evaluates a
  3×3 grid sweep (6×6/8×8/10×10 cells × 90/120/150 px) plus raw vs equalised
  preprocessing; `bench_eigenface` sweeps crop size (32/48/64 px) × retained
  eigenvalue energy (90/95/98/100 %) on top of the four metric/preprocessing
  variants. Both rank configurations by strict LOO rank-1, then margin, name the
  winner explicitly (including "shipped default wins/tied" language), and write the
  full sweep tables into the committed markdown reports.

### Changed — zero-dependency recognisers retuned on the harder gallery
- **LBPH default histogram grid changed from 8×8 to 6×6** (`LbphConfig` default;
  descriptor length 3 776 → 2 124 `f32`): the coarser grid wins at **every** swept
  crop size on the hard gallery (61–62/68 vs 59–60/68; shipped point 62/68 = 91.2 %
  rank-1) because larger cells pool the box-crop localisation jitter of an
  unaligned pipeline, with no regression on the easy gallery (33/33 for every
  grid/size combination). OpenCV's 8×8 remains one field away for landmark-aligned
  deployments.
- **`lbph::DEFAULT_MAX_DISTANCE` recalibrated from 30 to 16.7** for the new 6×6
  chi-square scale, deliberately as a conservative low-FAR point rather than an EER
  point: FAR 0.26 % / FRR 48.7 % at 96.5 % pair accuracy on the hard gallery
  (EER ≈ 22.5, FAR ≈ FRR ≈ 20 %), and FAR 1.2 % / FRR 16.5 % on the easy gallery.
  The same/different distributions overlap on hard data, so no threshold gives both
  low FAR and low FRR; the docs now direct close-set use to the threshold-free
  `rank_crop` API (91.2 % rank-1 is the honest headline).
- **Eigenfaces defaults kept** at 64 px / 98 % energy after the 16-point sweep showed
  58/68 = 85.3 % is the classical-PCA ceiling on the hard gallery and the shipped
  point already sits on it (64 px / 95 % only ties within one probe). The 6.3 accept
  constant is re-documented as a conservative low-FAR point there (FAR 1.5 %,
  FRR 51.8 %) while remaining the EER-region point on the easy gallery
  (FAR 2.9 % / FRR 3.5 %, rank-1 33/33).
- Recognition docs (`docs/recognition-lbph.md`,
  `docs/recognition-eigenface.md`, `docs/bench-results-*.md`, README) regenerated for
  the 77-crop / 21-identity numbers, sweep tables, and the threshold/rank guidance.
### Added — Swiss-army-knife sweep (SDK discoverability + uniform trait)

- **`FaceDetector` uniform-interface parity.** New `HaarDetector` newtype
  (`src/face_detector.rs`) adapts the crate-root Haar cascade
  `rsface::Detector` so it implements `FaceDetector`. Every algorithm the CLI
  accepts (`haar`, `cnn`, `yunet`, `mtcnn`, `hog`, `luminance`, `scrfd`) now
  implements the same trait and can be dispatched through
  `Vec<Box<dyn FaceDetector>>`. Fixes a real consistency gap.
- **CLI discovery flags.** `rs-face --list-algos` prints every algorithm with
  its maturity badge (`Production` / `Scaffold`) and description;
  `--list-features` prints every Cargo feature the binary was compiled with;
  `--version` prints the crate version. `--help` now ends with a RECIPES
  section showing 6 concrete invocations.
- **`docs/INDEX.md`** (new): single table-of-contents for every doc in the
  repo. Replaces the implicit "read README then poke around" workflow.
- **Four new SDK examples** (`cargo run --example <name>`):
  - `detect_haar` — smallest end-to-end Haar run on a synthetic frame
  - `detect_uniform` — "swiss-army-knife" demo: dispatch Haar + HoG via
    `Box<dyn FaceDetector>` heterogeneous vec
  - `recognise_lbph` — enrol + identify with zero-dep LBPH
  - `recognise_eigenface` — train + identify with zero-dep PCA
- **README rewrite.** New opening pitch ("the face-library swiss army knife"),
  what's-in-the-box matrix, 5-minute start, "pick the right algorithm" guide,
  Cookbook, Documentation index. Old duplicate sections (Algorithm diagram,
  Quick start, Architecture) removed; everything they covered is now in
  `docs/INDEX.md` + Cookbook.
- **Library-level docs (`//!` block in `src/lib.rs`)**: rewrote from a 9-module
  list to cover all 25+ public modules, plus a feature matrix, Maturity table,
  `no_std` notes, and SDK cookbook recipes (5-line examples for detection,
  uniform dispatch, recognition, pipeline).
- 4 new doctests added to `src/lib.rs`; **270 tests pass**, 0 fail.


### Added — zero-dependency recognition (LBPH)
- **LBPH face recogniser in the default build** (`src/lbph.rs`): radius-1 /
  8-neighbour LBP codes, OpenCV-compatible 59-bin uniform-pattern mapping,
  8×8 spatial L1-normalised histograms on 120 px crops, chi-square distance,
  multi-shot enrolment with best-member scoring, verification, identification
  with `max_distance`/`min_margin` policy (`LbphMatch`), and 9 unit tests. No
  weights download and no third-party crate.
- **Measured real-face accuracy** (`docs/recognition-lbph.md`,
  `docs/bench-results-lbph.md`): on 35 ArcFace-labelled drama-frame crops (8
  identities, 85 same / 510 different pairs) leave-one-out rank-1 is 33/33 over
  repeated identities, pair accuracy is 97.6 % at the calibrated zero-FAR
  default distance 30, EER ≈ 9.5 % near distance 48.
- `bench_lbph` bin (default features): pair-distance distributions, best
  threshold / EER / FAR-FRR, and rank-1 identification over a labelled crop
  tree; writes the markdown report.
- `prep_lbph_crops` bin (`required-features = ort-backend`, offline labelling
  only): SCRFD detection + ArcFace embedding clusters real frames into
  identities and writes plain box crops the zero-dep bench consumes.
- `tools/lbph_prep.sh`: JPG→PPM conversion (Pillow, lab-only), offline
  ArcFace labelling, zero-dep LBPH evaluation in one command.
- Documented honestly: the recogniser is accurate given a correct box, but the
  Haar cascade shipped in the default build is a demo cascade and does not
  select real faces on the drama frames — production zero-dep detection needs a
  trained `.rfcf` cascade (`--cascade`, `tools/convert_opencv_xml.py`).

### Added — zero-dependency recognition (Eigenfaces / PCA)
- **Eigenfaces recogniser in the default build** (`src/eigenface.rs`): the
  Turk–Pentland (1991) PCA recogniser with no weights and no third-party crate —
  64×64 vectorised crops, mean subtraction, the n×n Gram-matrix trick, and a
  pure-`std` cyclic-Jacobi symmetric eigendecomposition; 98 % eigenvalue-energy
  truncation (cap 255), Euclidean (default) and √λ-whitened Mahalanobis
  metrics, multi-shot best-member matching, verification, and the same
  `Match`/`BelowThreshold`/`Ambiguous`/`NoCandidates` policy as LBPH, with 7
  unit tests (Jacobi correctness, degenerate galleries, boundary probes).
- **Measured real-face accuracy** (`docs/recognition-eigenface.md`,
  `docs/bench-results-eigenface.md`): on the same 35 ArcFace-labelled crops
  (8 identities) under **strict leave-one-out with a full PCA retrain per
  probe**, rank-1 is 33/33 for the shipped Euclidean/raw variant; pair accuracy
  is 97.0 % at the calibrated EER-region default distance 6.3 (FAR 2.9 %, FRR
  3.5 %). The same/different distributions overlap (closest impostor 3.90 vs
  farthest genuine 7.40), so no usable zero-FAR point exists on this set —
  stated honestly, and the threshold is documented as deployment-dependent.
- `bench_eigenface` bin (default features): strict-LOO evaluation of all four
  preprocessing/metric variants, pair distributions, EER / best-threshold /
  crate-default operating points, and rank-1; writes the markdown report.
- Shared bench helpers extracted to `src/bin/bench_common.rs` (crop loading,
  EER/best-threshold sweeps, percentile stats), now used by both `bench_lbph`
  and `bench_eigenface`; `tools/lbph_prep.sh` runs both benches.

### Changed — zero-dependency recognition (LBPH)
- `Cargo.toml`: `autobins = false`; the `src/bin` targets are registered
  explicitly (now seven, incl. `bench_eigenface`) so the ort-gated prep bin
  stays out of the default build.

### Added — industrial recognition (SCRFD + ArcFace via ONNX)
- **SCRFD-10G face detector** (`src/scrfd.rs`, `src/scrfd_detector.rs`):
  letterbox preprocessing, anchor-free stride-8/16/32 decoding (no half-stride
  offset, matching InsightFace exactly), keypoint decoding, and post-processing.
  0.954 / 0.940 / 0.828 WIDER FACE easy/medium/hard AP upstream.
- **ArcFace R50 recognizer** (`src/arcface.rs`, `src/arcface_recognizer.rs`,
  `src/align.rs`, `src/embedding.rs`, `src/face.rs`): 5-landmark similarity
  transform alignment, 512-d embeddings, enrollment/identification gallery,
  and cosine matching with a configurable threshold.
- **Two real ONNX inference backends** (`src/onnx/`), both optional so the
  default build stays zero-dependency:
  - `ort-backend` — ONNX Runtime via dynamic loading (`ORT_DYLIB_PATH`), the
    route to GPU (CoreML / CUDA / TensorRT / DirectML / ROCm behind separate
    features).
  - `tract-backend` — pure-Rust CPU inference for single-static-binary deploys.
  - All 16 real-model integration tests pass under either backend.
- **Model registry with pinned SHA-256 digests** (`src/models.rs`): weights are
  verified before inference, with a zero-dependency SHA-256 implementation and
  typed licence tiers (research-only vs Apache-2.0) queryable at runtime.
- `tools/fetch_models.sh` to download YuNet (Apache-2.0) and the buffalo_l
  bundle (SCRFD + ArcFace, research-only) with digest verification.
- `bench-accuracy` bin (`benches/accuracy.rs`): end-to-end latency and
  same/different-identity cosine distributions on real weights; writes
  `docs/bench-results.md` and prints threshold guidance.
- Real-face PPM fixtures (`lena`, `biden`, `two-people`) and
  `tests/real_model_e2e.rs` — 16 integration tests covering integrity gates,
  geometry, blank-frame quietness, gallery round trips, discrimination, and
  scale invariance.
- Placeholder-weight detectors (`mtcnn`, `yunet`, `hog`) now report
  `Maturity::Scaffold` and are visibly labelled as detecting nothing.

### Changed
- README gains measured accuracy tables, a licences section (research-only vs
  commercial), and instructions to reproduce the numbers.
- `docs/benchmarks.md` documents both the InsightFace default threshold (0.36,
  TAR@FAR=1e-4) and the measured distribution midpoint (≈0.53 on the repo
  fixtures), with the formula for choosing your own.

### Added
- End-to-end face detection run on a 1080×1920 / 24 fps drama clip (ReelShort
  `reelshort_65da5136e047484c61021f92_63lsksowyg`, ~2:14, HEVC). Result:
  **626 / 626 frames with face (100 %)**, 786 total detection boxes,
  ~9.2 fps end-to-end on a 6-core aarch64 box.
- `docs/samples/` directory with three sample annotated PNGs from that run
  (early / mid / late frames).

### Documentation
- `CHANGELOG.md` (this file).
- `README.md` "Latest run" section linking to the sample frames and listing
  the run's measured throughput / detection counts.

## [0.1.1] — 2026-09-20 — Platform hardening round 2

Closes the deferred audit items from PR #4 plus a second security review
of the result. Items roll up by area; see PRs #4 / #7 / #8 / #10 for the
per-commit attribution.

### Added
- **`platform/server/src/zip.rs`** — dependency-free STORE-method ZIP writer
  (local headers + central directory + EOCD + const-built CRC-32 table).
  Used by `/api/jobs/{id}/download.zip` to package one job's annotated
  frames + face crops + `manifest.json` as a single archive.
- **`/api/jobs/{id}/compare`** TB (test-button) trigger wired through the
  existing compare pipeline (frontend `platform/web/compare.js`); governed
  polling (`visibilitychange`-aware).
- **`/api/health/deep`** endpoint — surfaces S3 + Postgres reachability
  with a 200 / 503 split; the shallow `/api/health` stays a no-IO probe
  for the Docker healthcheck.
- **`platform/server/src/lib.rs`** — server crate exposed as a library
  (`pub mod api / config / jobs / metrics / persist / s3 / zip`) so
  integration tests can import modules.
- **`platform/tests/integration.rs`** — 6 in-process integration tests
  with full HTTP round-trip via reqwest: deep_health,
  upload_image_end_to_end, ssrf_bypass_payloads_all_rejected (5 payloads),
  download_zip_returns_valid_archive (LOCAL + EOCD signature check),
  range_request_returns_206, available_algos_includes_expected_three.

### Security
- **SSRF parser-differential fixes** (`platform/server/src/api.rs`):
  - `normalize_host_for_check` peels percent-encoded (`%31%32%37.0.0.1`),
    IPv4-mapped IPv6 (`::ffff:127.0.0.1`), and rejects IDN / non-ASCII /
    `..` outright before SSRF classification.
  - `is_blocked_host` recognises `[::ffff:127.0.0.1]` and bare `::ffff:127.0.0.1`
    directly (defense in depth — even if normalize regresses, the IPv4
    rules still fire).
  - Numeric IPv4 (`2130706433`, `0x7f000001`, `0177.0.0.1`, `0xa9.0xfe.0xfe.0xfe`)
    now decodes to octets before classification.
  - `import_video_url` and `start_stream` both feed `url_authority_host`
    (strips userinfo / port) before SSRF classification.
- **Telemetry filter** is now case-fold + extended with `secret`,
  `password`, `token` (low #10).
- **Migrations**: `schema_migrations` bookkeeping, per-file transaction,
  lexer-aware `split_sql_statements` (nested block comments, line
  comments, quoted strings, `$tag$` dollar quotes). Fail-fast at startup
  on migration error.
- **`/api/health/deep` probes** bounded by 2 s `tokio::time::timeout` so a
  hung backend doesn't pin `spawn_blocking` (DoS hardening).
- **`/api/health/deep` body** collapsed `ok/fail/disabled` into a single
  boolean per backend, so anonymous callers can't infer whether the
  deployment runs with Postgres or in-memory mode (info-disclosure
  hardening).

### Performance / resource hygiene
- **Streaming S3 PUT** (`put_object_file`, `UNSIGNED-PAYLOAD` SigV4): the
  original media upload no longer `fs::read` + `to_vec` a multi-GB video
  into a Vec before sending.
- **Streaming `/media` Range responses** (`S3RangeStream`,
  `tokio_util::io::ReaderStream`, `File::take(len)`): scrubbing a 2 GB
  video in the browser no longer allocates 2 GB in the server.
- **Streaming `download_zip`** via `ZipWriter::finish_into<W: Write>` +
  `mpsc::Sender<Vec<u8>>` (depth 4) + `Body::from_stream`: peak memory
  is bounded by `4 × chunk-size`, not the full archive. 256 MiB cap
  still applies on top.
- **ffmpeg video import** gets `-fs` size cap + 600 s wall-clock timeout
  (`run_ffmpeg_wait_with_timeout`). Image-conversion twins get the same
  30 s timeout pattern (`run_ffmpeg_with_timeout_blocking`).
- **ureq split timeouts** (`timeout_connect` 10 s, `timeout_read` 600 s,
  `timeout_write` 600 s); large GETs no longer killed by the old single
  120 s budget.
- **`compare_algos`** refuses non-image jobs and > 16 MiB originals
  (local: `tokio::fs::metadata` gate; S3: bounded `get_object_range(0, +16 MiB)`
  so HEAD→GET TOCTOU can't grow past the cap).

### Observability
- **`tracing` + `tracing-subscriber`** added; `init_tracing()` in
  `main.rs` defaults to `info,rsface_platform=info,tower_http=info,axum=info`
  via `EnvFilter`. ~50 scattered `eprintln!` / `println!` calls in
  `persist.rs` / `jobs.rs` / `api.rs` / `main.rs` migrated to
  `tracing::warn!` / `tracing::info!` / `tracing::error!`. FATAL startup
  messages now log at `error!` so journald picks them up correctly.

### Correctness
- **`Job.worker` `JoinHandle`** (`jobs.rs`): `cleanup_job_media_blocking`
  joins the running worker thread (15 s deadline) before listing and
  deleting the S3 prefix — eliminates the delete-vs-write race that
  produced orphan S3 objects on rapid cancel/delete.
- **Poison tolerance** on every per-job `Mutex` (registry-wide `jobs`
  HashMap stays strict). Worker panic no longer wedges `/api/jobs` or
  `/metrics`.
- **`O(n²)` crop counting** replaced by a single running counter in the
  video loop.
- **Pre-permit queued cancel** writes `Cancelled` to PG so historical rows
  reflect the true terminal state (low #14).
- **`DATABASE_URL` set + connect fail** now `exit(1)` at startup instead
  of silently running memory-only and losing all persistence (low #13).

### Deployment
- **Multi-arch image build** (`platform/Dockerfile`): `TARGETARCH` maps to
  `aarch64` / `x86_64-unknown-linux-musl`; cross-arch cross-compile
  fails explicitly.
- **rustfs / postgres** pinned (`rustfs/rustfs:1.0.0`,
  `postgres:16.15-alpine`); console / DB ports bind `127.0.0.1` only
  (no LAN exposure); Postgres password parameterised via `.env`.
- **Dockerfile cascade path synced** with the `src/weights/` move
  (PR #7); cache-prefetch layer stubs cargo target placeholders so
  per-algorithm `required-features` trimming still resolves.

### Tests
- 57 platform unit tests pass (was 45 at PR #4; +12 from this round):
  numeric IPv4 SSRF, IPv4-mapped / percent-encoded SSRF, streaming
  Range, `/media` path-traversal, queued cancel, `local_range_stream`,
  s3 list / SigV4 canonical-query, header round-trip, JSON cache, config,
  zip layout + CRC vectors, `key_extension_sanitizes_path_traversal`, etc.
- 6 platform integration tests pass (Colima docker-compose live):
  full HTTP round-trip for upload / detect / delete / SSRF / Range / zip.
- 17 GitHub Actions checks green: 13-combination feature matrix on the
  core crate + core on ubuntu + macos + platform on ubuntu.

### Live verification
Five SSRF bypass payloads were rejected by the live stack on the merged
main (Colima VM docker-compose):

```
rtsp://[::ffff:127.0.0.1]/c   → 400 url host is blocked
http://user:pass@127.0.0.1/x  → 400 url host is blocked
http://%31%32%37.0.0.1/x      → 400 url host is blocked
rtsp://0x7f000001:554/cam     → 400 url host is blocked
rtsp://2130706433/cam         → 400 url host is blocked
```

## [0.1.0] — 2026-08-13

### Added
- Initial release: zero-dependency Viola-Jones face detector in pure Rust.
- Five-feature-family Haar cascade, OpenCV `haarcascade_frontalface_default.xml`
  loader via the included `.rfcf` binary format and XML→`.rfcf` converter.
- 24×24 CNN detector (Conv→ReLU→Pool→FC→Sigmoid) with bundled placeholder
  weights so the CNN code path is exercisable without external downloads.
- Multi-threaded pipeline (`source → N detector workers → sink`) with
  greedy NMS, optional OpenCL squared-integral pre-filter, and a hand-rolled
  JSON manifest + annotated PNG writer.
- Frame sources: image sequence, HTTP, ffmpeg pipe, synthetic test pattern.
- CI workflow (`.github/workflows/ci.yml`): build + test on Ubuntu and macOS,
  `cargo fmt --check`, `cargo clippy -- -D warnings`.

### Notes
- `Cargo.lock` is committed because this crate ships as a binary-first
  project. Consumers should treat the lockfile as informative when depending
  on the library.
- Tilted (45°) Haar features evaluate to 0 (rotated integral kept correct
  under Rust's borrow checker with zero deps is non-trivial; the cascade
  format still supports them).
- CNN bundled weights are placeholders — train on labelled data and load
  via `CnnDetector::with_weights` for production use.
