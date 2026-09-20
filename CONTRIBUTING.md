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

The full tree — every directory at the repo root and what it is for. **Do
not add a new top-level directory without updating this table.** Every
file in the repo should have exactly one natural home; if you can't find
it, that's a smell worth fixing before the change lands.

```
rs-face/
├── Cargo.toml          # zero-dep package metadata (workspace-free, single crate)
├── Cargo.lock          # committed (binary-first project — reproducible builds)
├── LICENSE             # MIT
├── README.md           # public-facing overview, accuracy tables, quick start
├── CHANGELOG.md        # release notes (Keep a Changelog format)
├── CLAUDE.md           # agent rules (docker-only platform, pnpm dev frontend, etc.)
├── CONTRIBUTING.md     # this file
├── clippy.toml         # lint thresholds
├── rustfmt.toml        # formatting config
├── .github/workflows/  # CI: build, test, fmt, clippy (ubuntu + macOS matrix)
│
├── src/                # library + CLI source (Rust)
│   ├── lib.rs          # library entrypoint, declares every pub mod
│   ├── main.rs         # CLI binary (rs-face)
│   ├── align.rs        # landmark-based 112×112 face alignment (ArcFace prep)
│   ├── arcface.rs      # ArcFace pre/post-processing (ONNX-input agnostic)
│   ├── arcface_recognizer.rs # end-to-end ArcFace matcher
│   ├── detector.rs     # multi-scale sliding window + NMS
│   ├── eigenface.rs    # zero-dep PCA / Turk-Pentland recogniser
│   ├── embedding.rs    # FaceEmbedding + Gallery matcher
│   ├── face.rs         # Detection, Face, landmark types shared across modules
│   ├── face_detector.rs# the uniform FaceDetector trait
│   ├── hog_face.rs     # HOG + Linear SVM detector
│   ├── integral.rs     # integral + rotated + squared integral images
│   ├── lbph.rs         # uniform-LBP histograms + chi-square distance
│   ├── luminance_face.rs # band + symmetry detector (no weights)
│   ├── models.rs       # model registry with SHA-256 pinned digests
│   ├── mtcnn.rs        # P-Net → R-Net → O-Net cascade
│   ├── output.rs       # PNG / JSON manifest writers
│   ├── pipeline.rs     # multi-threaded pipeline (source → N detectors → sink)
│   ├── scrfd.rs        # SCRFD pre/post-processing (InsightFace)
│   ├── scrfd_detector.rs # end-to-end SCRFD detector
│   ├── video_id.rs     # cross-video tracker + clusterer + re-id
│   ├── yunet.rs        # YuNet-style anchor-based detector
│   ├── haar/           # features, cascade, demo cascade, OpenCV XML→.rfcf
│   ├── cnn/            # 24×24 Conv→ReLU→Pool→FC→Sigmoid CNN detector
│   ├── gpu/            # GPU backend trait + metal/cuda/rocm/mlu/ascend/cpu impls
│   ├── image/          # PNG / PPM codec, GrayImage / RgbImage
│   ├── onnx/           # backend-agnostic forward pass (ort vs tract)
│   ├── pool/           # worker pool helper used by the pipeline
│   ├── source/         # FrameSource trait + ffmpeg / http / image-seq impls
│   ├── weights/        # placeholder *.bin weights for include_bytes! scaffolds
│   └── bin/            # extra CLI binaries (see below)
│
├── src/bin/            # explicit [[bin]] targets (Cargo.toml sets autobins=false)
│   ├── bench_common.rs # #[path]-included helper shared by bench_lbph + bench_eigenface
│   ├── bench_detect.rs # micro-bench the Haar detector
│   ├── bench_lbph.rs   # gallery accuracy bench for LBPH
│   ├── bench_eigenface.rs # gallery accuracy bench for eigenfaces
│   ├── cnn_train.rs    # tiny CNN training driver
│   ├── debug_cascade.rs# dump a `.rfcf` cascade's structure
│   ├── prep_lbph_crops.rs # offline ArcFace crop pre-processing (ort-backend only)
│   └── rs_face_detect.rs # alternate CLI with GPU dispatch
│
├── tests/              # integration tests + ignored benchmark runners
│   ├── bench_components.rs   # micro-bench individual components
│   ├── bench_detect.rs       # larger-scale detection bench
│   ├── real_model_e2e.rs     # end-to-end with real ONNX models
│   ├── regression_dump.rs    # snapshot detector output for regression diff
│   ├── video_id_e2e.rs       # cross-video re-id round trip
│   ├── e2e_stress.sh         # bash stress script (run manually)
│   └── fixtures/             # small reference images used by the above
│
├── benches/            # criterion-style benches (harness = false)
│   ├── perf_compare.rs # CPU vs GPU perf shootout
│   ├── accuracy.rs     # gallery accuracy sweep (ort-backend only)
│   └── RESULTS.md      # human-readable bench summaries
│
├── examples/           # cargo run --example <NAME> (public cookbook)
│   ├── cascade_dump.rs           # parse a `.rfcf` and print its structure
│   ├── detect_haar.rs            # smallest end-to-end Haar run on a synthetic frame
│   ├── detect_scrfd_arcface.rs   # survey the ONNX model registry
│   ├── detect_uniform.rs         # trait-dispatch demo across Haar + HoG
│   ├── identify_short_drama.rs   # end-to-end video-level re-id (see docs/recognition-video.md)
│   ├── lena_classify_stages.rs   # walk a real cascade stage by stage
│   ├── recognise_eigenface.rs    # train + identify with eigenfaces, no weights
│   ├── recognise_lbph.rs         # enrol + identify with LBPH, no weights
│   └── synthetic_smoke.rs        # zero-data smoke test
│
├── docs/               # long-form design + accuracy + format documentation
│   ├── INDEX.md        # entry point; read this first
│   ├── architecture.md   # crate map + pipeline plumbing
│   ├── algorithms.md   # per-algorithm reference + picker matrix
│   ├── benchmarks.md   # reproducible bench scripts
│   ├── BENCHMARK_BASELINE.md # pinned regression baselines
│   ├── bench-results*.md # measured accuracy / throughput tables
│   ├── CASCADE_FIX.md  # history of OpenCV XML parser bugs
│   ├── CPU_VS_GPU_REPORT.md # when GPU helps, when it doesn't
│   ├── format.md       # `.rfcf` binary cascade format
│   ├── GPU_BACKENDS.md # CPU / Metal / CUDA / ROCm / MLU / Ascend
│   ├── recognition-*.md # per-recogniser deep dives
│   └── samples/        # annotated PNGs from real detection runs
│
├── tools/              # Python + shell helpers (not part of the Rust crate)
│   ├── convert_opencv_xml.py  # canonical OpenCV XML → .rfcf converter
│   ├── xml_to_rfcf.py         # legacy variant (same job, kept for history)
│   ├── convert_res10_to_onnx.py # res10 SSD → ONNX weight conversion
│   ├── fetch_models.sh        # pinned ONNX downloader (Y/N prompt)
│   ├── lbph_prep.sh           # Pillow-based JPG→PPM crop prep for benches
│   ├── crop_faces.py          # extract aligned face crops from a folder
│   ├── annotate_all_faces.py  # overlay detections on every image in a folder
│   ├── extract_top_faces.py   # keep only the N highest-scoring crops
│   ├── detect_gpu.py          # standalone GPU detection runner
│   ├── compare_cpu_gpu.py     # side-by-side CPU vs GPU result diff
│   ├── parallel_detect.py     # multi-process batch detector
│   ├── gpu_backends.py        # probe available GPU runtimes
│   └── run_rust_detect.py     # shell out to `cargo run --bin rs_face_detect`
│
├── platform/           # deployment-grade server + web UI (Docker-only)
│   ├── Cargo.toml      # platform-server crate (separate from the library)
│   ├── Cargo.lock
│   ├── Dockerfile      # rsface-server image (non-root user, ffmpeg, static binary)
│   ├── Dockerfile.gpu  # CUDA-enabled variant
│   ├── docker-compose.yml # 3-service stack (rustfs + postgres + rsface-server)
│   ├── docker-compose.gpu.yml # GPU-equipped variant
│   ├── README.md       # port map (20080/19000/15432) + ops overview
│   ├── DOCKER.md       # authoritative docker guide (deploy/backup/restore/...)
│   ├── DOCKER_SIZING.md # recommended instance sizes per workload
│   ├── MINIMUM_CONFIG.md # smallest viable box
│   ├── PROFILE.md      # performance profile from real load tests
│   ├── CHANGELOG_CNN.md # platform CNN-specific history
│   ├── CHANGELOG_PERF.md # perf-pass history (cache/gzip/ETag)
│   ├── docs/           # platform design + roadmap + SDK notes
│   ├── migrations/     # SQL migrations (0001_init, 0002_telemetry, ...)
│   ├── scripts/        # bash + node smoke / screenshot scripts
│   ├── server/         # axum HTTP API + worker pool
│   │   ├── src/
│   │   │   ├── main.rs       # binary entrypoint
│   │   │   ├── api.rs        # all HTTP handlers
│   │   │   ├── cache.rs      # zero-dep TTL cache
│   │   │   ├── config.rs     # env + CLI config
│   │   │   ├── jobs.rs       # in-process job queue
│   │   │   ├── metrics.rs    # Prometheus /metrics endpoint
│   │   │   ├── persist.rs    # Postgres adapter
│   │   │   ├── s3.rs         # rustfs / S3 adapter
│   │   │   └── bin/s3test.rs # S3 round-trip smoke binary
│   │   └── Cargo.toml
│   ├── testdata/       # reference inputs (small images committed; mp4/HLS gitignored)
│   └── web/            # vanilla-JS frontend (zero build step in container)
│       ├── index.html
│       ├── app.js          # main bundle
│       ├── toast.js        # multi-level toast helper
│       ├── modal.js        # confirmModal + modalKit + ctxMenu
│       ├── dashboard.js    # stats dashboard
│       ├── visibility.js   # tab visibility lifecycle helper
│       ├── compare.js      # algorithm compare mode (5 algos side by side)
│       ├── telemetry.js    # auto-captured page_view / api_call / js_error events
│       ├── style.css       # layout
│       ├── theme.css       # theme tokens (light/dark)
│       └── README.md       # frontend-local notes
│
├── package.json        # frontend dev (vite only); root: 'platform/web'
├── pnpm-workspace.yaml # pnpm config (onlyBuiltDependencies allow-list)
├── pnpm-lock.yaml      # committed for reproducible dev installs
├── vite.config.js       # vite dev server with /api + /events proxy → :20080
│
└── data/               # bind-mounted runtime state (gitignored; created by docker compose)
    ├── rustfs/         # object store contents (jobs/, frames/, thumbnails/, ...)
    ├── pg/pgdata/      # postgres data directory
    ├── pg/rsface_dump.sqlc # optional PG backup (made by pg_dump --schema-only ...)
    └── media/          # raw uploads + extracted frames mirror
```

### Conventions

- **Top-level directories are sparse by design.** Adding `core/`, `misc/`,
  `scratch/`, `notes/`, or anything similar is almost always a smell —
  either the contents belong in `src/`, `tests/`, `examples/`, `docs/`,
  `tools/`, or `platform/`. If you need a scratch space, use `/tmp/` or
  `/scratch/` (both already gitignored).
- **Working planning documents do not belong at the repo root** — they're
  noise on every `git status` and they pin internal scratch decisions into
  the public history. Use your local branch + commit history for those;
  merge them into `CHANGELOG.md`, `docs/`, or `TASK_PLAN.md` lives in
  your editor, not the repo.
- **`src/bin/` is only for `[[bin]]` targets.** Files that exist only to
  be `#[path]`-included by other bins (currently `bench_common.rs`) keep
  their position in `src/bin/` so the relative include path stays a
  single filename, but they MUST NOT be reachable as a standalone bin —
  `autobins = false` plus the file's own header docstring enforces this.
- **`platform/web/` is zero-build.** No bundler, no transpiler, no
  `node_modules` in production. Each `<script>` in `index.html` is a
  classic defer script loaded directly by the browser.
- **`platform/testdata/` is split:** small reference images (`*.jpg`,
  `*.pgm`, `*.ppm`) are checked in; large binaries (`*.mp4`, HLS loop
  segments) are gitignored and downloaded on demand.
- **`data/` is gitignored and ephemeral.** It's created by
  `docker compose -f platform/docker-compose.yml up -d --build` and lives
  as a bind mount under the repo root so backups are `rsync`-friendly.
  Don't commit anything under `data/`.
- **No Makefile by convention.** Build / dev / deploy commands are inline
  cargo / docker compose / pnpm invocations in the docs, or shell scripts
  under `platform/scripts/` and `tools/`. Don't reintroduce a Makefile.

### Module split

The module split is intentional: each top-level module should compile
in isolation and have minimal cross-deps beyond `image` and `integral`.
New algorithms belong behind the [`FaceDetector`] trait
(`src/face_detector.rs`) so the pipeline can dispatch them uniformly.

## Coding conventions

- `cargo fmt` for formatting (CI enforces `--check`).
- `cargo clippy --release --all-targets -- -D warnings` (CI enforces).
- The library is **zero-dep**. Do not add a `Cargo.toml` `[dependencies]`
  entry without prior discussion — the headline feature is the absence of
  one.
- Hand-rolled parsers / decoders are fine and encouraged; pulling in a crate
  for a 50-line codec is not.

## Working with the platform services (Docker)

The platform server (`platform/server/`) and its sidecars (rustfs, postgres) are
**only** run via Docker. Don't try `cargo run` for `rsface-server` — it needs
ffmpeg + an S3 endpoint + a running Postgres, all of which compose handles.

```bash
docker compose -f platform/docker-compose.yml up -d --build   # start rustfs + postgres + rsface-server
docker compose -f platform/docker-compose.yml logs -f --tail=100   # tail logs from all 3 services
docker compose -f platform/docker-compose.yml ps              # container status + health
bash platform/scripts/docker-smoke.sh                          # e2e smoke (/api/health + image upload + PG count) — needs the stack up first
docker compose -f platform/docker-compose.yml down             # stop (keeps data/ bind mounts intact)
docker exec -i rsface-postgres pg_restore \
    -U rsface -d rsface --clean --if-exists --no-owner --role=rsface \
    < data/pg/rsface_dump.sqlc                                # restore PG from data/pg/rsface_dump.sqlc
```

Data lives in `data/{rustfs,pg/pgdata,media}/` (bind-mounted from the repo root,
visible and rsync-friendly). See [`platform/DOCKER.md`](platform/DOCKER.md)
for the full ops guide — backup / restore / migrate-from-old-named-volume /
troubleshoot / cleanup.

## Frontend development (pnpm dev + hot reload)

The frontend (`platform/web/`, ~5,000 lines of zero-dependency vanilla JS) is
served by the Rust server in production, but for iteration we run it under
**Vite** with a proxy to the Docker backend, so any edit to `app.js` /
`index.html` / `style.css` is reflected instantly via Vite HMR:

```bash
docker compose -f platform/docker-compose.yml up -d --build   # backend must be running first
pnpm dev                                                       # vite dev server on http://localhost:5173/ → proxies /api to :20080
```

Vite root is `platform/web/`; proxy routes `/api/*` and `/events` to
`http://localhost:20080`. **Don't** edit the Docker backend's bundled
`/app/web/` — your changes will be lost on the next `docker compose up -d
--build`. Always edit `platform/web/` on the host.

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