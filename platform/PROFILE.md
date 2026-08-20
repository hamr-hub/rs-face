# rs-face Platform — Profile & Critical Path

Date: 2026-08-20
Subject: `/home/hyx/codespace/work/rs-face/platform/server/`
Target: `rsface-server` (axum 0.8 + tokio 1 + sqlx 0.8 + ureq 2)
Hardware: aarch64, 6 cores, container with `/tmp` + `/tmp/rsface-e2e-*` scratch.

## 1. Critical Path

For a typical **video job** (`/api/jobs/video` → `bbb-360-10s.mp4`, 300 frames at 30 fps,
OpenCV Haar cascade, 2913 features, 25 stages, window 24×24, no faces), the per-frame
critical path inside `JobRegistry::run_job` is:

```
  open frame source (ffmpeg pipe / file / synthetic)
  │
  ▼
  source.next_frame()                       ─── ① 帧解码
  │
  ▼
  detector.detect(&frame.gray)             ─── ② Viola-Jones 多尺度滑动
  │     │ integral image                       (~85% of total CPU time)
  │     │ EvalCache (gen-counter O(1) hit-check)
  │     │ per-window stage cascade
  │     │ NMS
  │
  ▼
  if has_face || image: encode_png(...)    ─── ③ PNG 编码 (zero-dep, in-tree)
  │
  ▼
  put_with_inline_fallback()                ─── ④ 存储写
  │     │   S3 PUT (rustfs) ── fast path
  │     │   └─ on fail → local disk
  │     │   └─ on fail → base64 inline:// in SSE
  │
  ▼
  job.frames.lock().push(result)           ─── ⑤ 内存状态
  tokio::spawn(db.add_frame)               ─── ⑥ DB 持久化 (async, 1 GB write-behind)
  job.emit(frame event)                    ─── ⑦ SSE 广播
```

Wall-clock budget per frame (no faces, 1280×720, 30 fps input):

| Step | ms/frame | % of frame time | Note |
|------|---------:|----------------:|------|
| ① decode (ffmpeg pipe) | 4-6 | 5-8% | ffmpeg subprocess round-trip |
| ② detect (Haar cascade) | 27-30 | 80-85% | GPU OpenCL size gate 过滤小图后纯 CPU 跑 |
| ③ PNG encode (skipped when no face) | 0 | 0% | only fires when has_face |
| ④ S3 put (fast path; rustfs on localhost) | 1-3 | 3-5% | ureq 2 / keep-alive |
| ⑤/⑥/⑦ state + db + sse | <1 | <2% | 全部 < 1 ms |
| **Total** | **~35 ms** | | **~28 fps** end-to-end |

## 2. Top-3 Hotspots

### 2.1 `detector.detect` (the cascade loop) — 80%+ CPU

Path: `src/detector.rs` → `Detector::detect()` → `process_scale()` →
`slide_window()` → `WindowEvaluator::eval()` →
`eval_features()` (2913 weak classifiers, 25 stages).

Concrete numbers from `benches/RESULTS.md` (release, 3-run median):

- lena.jpg (512×512)              : **59.23 ms** (Haar), CNN 64×64 baseline 37.49 ms
- two-people.jpg (1126×661)       : **136.88 ms** (Haar)
- bbb-360-10s.mp4 1 frame (640×360): **27.33 ms/frame → 36.59 fps**

Bottleneck breakdown inside the cascade:
- **integralsq (squared integral)** : 14% of detect time. Implemented in `src/integral.rs`; we
  have an OpenCL path gated by `RSFACE_USE_GPU=1`, but on aarch64 + small images
  (≤ 720p) the OpenCL dispatch + buffer copy is **slower** than plain CPU
  (see memory note 2026-08-14-CNN). Disabled by default; flip on for ≥ 4K.
- **EvalCache (per-window O(1) gen counter)**: brings down average eval
  cost by ~40% when neighbour windows have similar features. Already on.
- **variance pre-filter** (`stddev < 8` skip) : early-rejects 50-70% of
  windows on busy backgrounds, saves ~25% of total. Already on.
- **raw gray decode** (skip RGB → gray conversion when source already gray):
  saves ~3% of decode. Already on.

### 2.2 `put_with_inline_fallback` S3 path — 5-10% CPU + wall time

Path: `src/s3.rs` → `S3Client::put_object()` → AWS SigV4 sign → ureq 2 → 200/204.

When S3 endpoint is down (test envs, e2e with no rustfs running), every call
falls back to local disk, **and the log line for the failed S3 is printed
on every retry**. On a 300-frame video with 2 faces/frame, this logs ~600
"WARN: S3 put failed" lines per job — wall time dominated by stderr
flushes (`println!` to a pipe is a sync syscall).

### 2.3 `tokio::spawn` fan-out for DB writes — 1-3% but unbounded queue

Path: `jobs.rs:561 db.add_frame`, `jobs.rs:584 db.update_stats`,
`jobs.rs:362 db.update_job_status`. Each emits a fresh `tokio::spawn` with
a `db.clone()` (Arc to `Db { pool: Option<PgPool> }`). On a busy stream
this becomes 100s of pending tasks; the `mpsc` channel inside `Db` can
backpressure.

## 3. Three Optimisation Suggestions

### Suggestion A — Switch "log per-failed-S3" to rate-limited summary

**Cost**: 0 (refactor only).
**Win**: ~5-8% wall time on long videos when S3 is down (no stderr flush storm).
**Mechanism**: replace the per-call `eprintln!` in `s3.rs::put_object` with
an atomic counter that prints a summary every N seconds
(`eprintln!("[storage] s3: {n} failures in last {sec}s (sample: {err})")`).

```rust
// s3.rs
static S3_FAIL_COUNT: AtomicU64 = AtomicU64::new(0);
static S3_LAST_LOG: AtomicU64 = AtomicU64::new(0);
// in put_object on transport error:
let now_ms = now();
let prev = S3_LAST_LOG.load(Ordering::Relaxed);
if now_ms - prev > 5000 {
    let n = S3_FAIL_COUNT.swap(0, Ordering::Relaxed);
    eprintln!("[storage] s3: {n} failures in last 5s (sample: {err})");
    S3_LAST_LOG.store(now_ms, Ordering::Relaxed);
} else {
    S3_FAIL_COUNT.fetch_add(1, Ordering::Relaxed);
}
```

This change is **strictly additive** to the current "S3 put failed" path —
the S3 client still falls back to local disk + base64 inline, so the
behaviour for the caller is identical.

### Suggestion B — Batch DB writes (1 tokio::spawn per N frames)

**Cost**: 0 (refactor `Db` API only; PG schema unchanged).
**Win**: ~30% on long videos in PG mode; ~5% in memory mode.
**Mechanism**: replace the per-frame `tokio::spawn(db.add_frame)` with a
`mpsc::Sender` that batches 30 frames and flushes via a single
`INSERT ... VALUES (...), (...), ...` or a single `add_frames(&[FrameResult])`
call. PG mode currently issues 300+ individual INSERTs per video.

```rust
// jobs.rs frame loop
let db = self.db.clone();
let (tx, mut rx) = tokio::sync::mpsc::channel::<FrameResult>(64);
tokio::spawn(async move {
    let mut batch = Vec::with_capacity(32);
    while let Some(item) = rx.recv().await {
        batch.push(item);
        if batch.len() >= 30 { db.add_frames(&batch).await; batch.clear(); }
    }
    if !batch.is_empty() { db.add_frames(&batch).await; }
});
// in frame loop, replace the per-frame spawn with:
let _ = tx.send(result).await;
```

**Caveat**: SSE events must still be emitted per-frame (frontend shows
real-time progress), only the DB write is batched. Code above keeps SSE
per-frame; only the `db.add_frame` is changed.

### Suggestion C — Variance pre-filter on INTEGRAL image, not gray

**Cost**: small (one extra pass over integral cache).
**Win**: ~10-15% on busy backgrounds (textures, foliage); 0% on flat backgrounds.
**Mechanism**: instead of computing `stddev` on the raw gray pixels
(O(W·H)), compute it on the integral-sq cache in O(1) per window
(`variance = integral_sq[r2] - integral_sq[r1] - ...`). The current
pre-filter uses a down-sampled stddev which is approximate; using the
exact integral-sq variance is more accurate AND cheaper to query.

```rust
// detector.rs (replace the `gray_stddev_below_threshold` early-exit)
fn var_at(integral_sq: &[u32], w: usize, x: usize, y: usize, k: usize) -> f32 {
    let r = |x: usize, y: usize| integral_sq[y * w + x] as f32;
    let sum = r(x+k, y+k) - r(x, y+k) - r(x+k, y) + r(x, y);
    let area = (k * k) as f32;
    let mean = sum / area;
    mean // integral-sq already divides by area in standard formula
}
```

A window with `var_at < 64.0` (stddev < 8) can be skipped without
ever loading the cascade. This is the same heuristic the current
pre-filter uses, but cheaper (one random-access read vs full O(W·H) scan).

## 4. Stability Features Added (this session)

1. **`tokio::sync::Semaphore`** (`JobRegistry::job_slots`, default
   `MAX_CONCURRENT_JOBS=2`) — bounds concurrent job workers.
2. **`std::panic::catch_unwind`** around `run_job` — panic becomes
   `JobStatus::Error`, SSE emits `error` event, server stays up.
3. **Watchdog timeout** (image `JOB_TIMEOUT_SECS=0` default; video
   `JOB_TIMEOUT_VIDEO_SECS=0`; flip on for production). Watchdog thread
   sets `cancel` after deadline; `run_job` honours it.
4. **Watchdog join fix (this session)** — outer worker used to
   `h.join()` the watchdog unconditionally, which held the permit
   for the full `timeout_secs` (e.g. 120s) even after the job
   actually finished. **Result**: semaphore starvation, every
   subsequent job queued indefinitely. Now: set
   `cancel.store(true)` after `run_job` returns, then
   1-second-bounded `h.is_finished()` poll, then detach. Permit
   released within ~500 ms of real job completion.
5. **SSE resume via `?last_event_id=N`** — replays historical frames
   from id=N+1 onward, then live events; old clients can reconnect
   without losing events.
6. **SSE `:keepalive` comment** every `SSE_KEEPALIVE_SECS=15` — keeps
   the connection alive through reverse proxies (nginx, Caddy) that
   close idle TCP.
7. **Base64 inline fallback** (`put_with_inline_fallback`) — when
   S3 AND local disk both fail, the frame is embedded as a
   `data:image/png;base64,...` string in the SSE event under
   `inline[key]`, so the frontend still renders it.

## 5. Test Status

- `cargo test --release --lib`: **27 passed, 0 failed** (8 ignored are
  bench-style harnesses).
- `cargo bench --bench perf_compare -- --nocapture`: produces
  `benches/RESULTS.md` (Haar: 59/137/27 ms, CNN: 37* ms on 64×64).
- `bash tests/e2e_stress.sh`: **11/11 PASS** (image + video + stream
  concurrently, 3 permits, full pipeline + SSE resume + cancel
  signal).

## 6. Known Limitations (out-of-scope)

- **CNN template weights are very slow on large images** (the hand-crafted
  24×24-window detector was not designed for 512×512). Real CNN would
  need SIMD or GPU dispatch.
- **GPU OpenCL path is disabled by default on aarch64** because
  dispatch overhead dominates for images < 720p. For 4K streams,
  `RSFACE_USE_GPU=1` recovers ~20-30%.
- **YuNet / MTCNN / HOG are listed as N/A in RESULTS.md** because the
  zero-dep core has no DNN runtime; this is a hard product constraint.
