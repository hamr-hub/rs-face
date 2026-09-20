# Changelog

All notable changes to `rs-face` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — stale docs after detector / GPU backend cleanup
- **`CONTRIBUTING.md`** — removed references to deleted source files
  (`src/hog_face.rs`, `src/mtcnn.rs`, `src/yunet.rs`), corrected the
  `src/gpu/` description (currently `metal/cuda/mod/backend.rs` behind
  features, not the 6-vendor matrix in CONTRIBUTING), and updated
  `src/weights/` (now bundled real OpenCV cascade, not placeholder
  `*.bin`s). Also clarified `examples/detect_uniform.rs` and
  `docs/GPU_BACKENDS.md` to match the current `haar/cnn/luminance`
  detector set and `Metal/CUDA` GPU set.
- **`docs/CPU_VS_GPU_REPORT.md`** — removed references to non-existent
  `src/gpu/rocm.rs`, `src/gpu/ascend.rs`, `src/gpu/mlu.rs` stubs and
  the stale "AMD / Mac / NVIDIA / 国产 GPU 全部接好" claim. Reality:
  Apple Silicon (Metal) and NVIDIA (CUDA, feature-gated) are the only
  actively supported paths; ROCm / Ascend / MLU / DirectML are
  intentionally off the roadmap.

## [Unreleased]

### Changed — code-quality sweep (no behaviour change)
- **`src/main.rs`** — `run_cnn_pipeline` and `run_algo_pipeline` were
  near-identical (~120 lines of duplicated frame-loop / RGB-fallback /
  record-building / manifest-write code). Replaced by a single shared
  `run_with_detector` driver; the CNN path now adapts its `&[f32]` /
  `CnnDetection` contract to the shared `Fn(&GrayImage) -> Vec<Detection>`
  signature via a small closure. Same manifest layout, same `--only-with-face`
  semantics, same `--out` paths.
- **`OnnxError::source()`** (`src/onnx/mod.rs`) now exposes the wrapped
  `io::Error` as the error source (matches the `LbphStoreError` /
  `SubspaceStoreError` pattern). Previously only `Display` carried the
  underlying message; `Error::source()` returned `None`.
- **`IdentifyError<D, E>::source()`** (`src/video_id.rs`) now surfaces
  the inner detect / embed error through `Error::source()`; previously
  the default empty impl hid them.

### Fixed — broken intra-doc links & stale cross-references
- 7 broken intra-doc links in the library (`IntegralTable`,
  `prefix_sums_fit_u32`, `DEFAULT_FRAMES`, `crate::linalg`,
  `identify_videos` → real `identify_video`, plus two `SquaredIntegralImage`
  / `Detector::detect` resolutions) and 4 in `src/bin/cnn_train.rs`
  (`[epochs]`/`[out_path]`/`[seed]` were being parsed as doc links; usage
  example is now wrapped in a `text` code block; `CnnDetector` now uses
  the qualified `rsface::cnn::CnnDetector` path).
- 4 doc warnings in `platform/server/` (`Vec<u8>` / `Vec<bool>` /
  `Arc<JobRegistry>` / `[0,1]` were being read as HTML tags or range
  links); now escaped with backticks.
- **`src/gpu/metal.rs`** — replaced a dangling `// FIXME` reference with a
  concrete note pointing at the existing `MetalBackend::detect_windows_cpu`.
- **`docs/GPU_BACKENDS.md`** — dropped the corresponding `FIXME` cross-link.
- **`src/pipeline.rs`** — removed the unused `#[allow(dead_code)]
  fn _ensure_grayimage_send` stub.

### Added — edge-case detector tests
- `detection_iou_nested_and_symmetric` (`src/detector.rs`) covers IoU for
  one box fully contained in another, partial overlap, and verifies
  `a.iou(b) == b.iou(a)`.
- `detector_with_empty_image_returns_no_detections` covers the
  `0×0`, `10×0`, `0×10` zero-area cases so future refactors cannot
  regress the empty-source path.

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
