# Video-level face identification

`rsface` ships three single-image recognisers (LBPH, eigenface, ArcFace) that
answer **"is this face the same identity as one I have on file?"**. The
[`video_id`](../src/video_id.rs) module answers the question those can't:

> Across one or many videos, **which physical people appear, and when?**

The pipeline is layered so each piece is independently testable:

```text
[video1.mp4, video2.mp4, ...]
        │
        ▼  source::open() — ffmpeg pipe, image seq, HTTP, or `test://`
frame (gray + rgb)
        │
        ▼  detector + recogniser per frame (any combination)
Vec<Detection> + Vec<Embedding>
        │
        ▼  LightTracker — IoU-based per-video track association
Vec<Track>     (one Track = a stable id, accumulating embeddings across frames)
        │
        ▼  IdentityCluster — single-linkage, cosine ≥ threshold
Vec<Identity>  (one Identity = a cluster_id, a centroid, the tracks that landed)
        │
        ▼  merge_across_videos — re-cluster across the whole batch
Vec<Identity>  (Alice-in-ep01 == Alice-in-ep03)
```

## What each piece does

| Layer | Source | Job |
|---|---|---|
| `LightTracker` | `src/video_id.rs:154` | IoU-greedy association. One frame's detections attach to existing tracks or open new ones; tracks retire after `track_max_age` frames without a match. |
| `IdentityCluster` | `src/video_id.rs:332` | Single-linkage clustering of unit embeddings with cosine ≥ `cluster_threshold`. Centroids are blended with each new mean so slow lighting / makeup drift doesn't lock an identity out. |
| `identify_video` | `src/video_id.rs:567` | Glue: source → tracker → clusterer → `VideoIdentification`. |
| `merge_across_videos` | `src/video_id.rs:625` | Re-runs single-linkage across the tracks of every per-video run so cross-episode re-id lands on one global `cluster_id`. |

## Zero-dep end-to-end

The example [`examples/identify_short_drama.rs`](../examples/identify_short_drama.rs)
wires `HaarDetector` + `LbphRecognizer::extract` (both zero-dep) into the
pipeline:

```text
$ cargo run --example identify_short_drama -- \
      clips/ep01.mp4 clips/ep02.mp4 clips/ep03.mp4 \
      --out ./out --algo haar
>>> processing clips/ep01.mp4
    ep01: 4320 frames, 1237 faces, 4 identities
>>> processing clips/ep02.mp4
    ep02: 4200 frames, 1180 faces, 3 identities
>>> processing clips/ep03.mp4
    ep03: 4180 frames, 1310 faces, 4 identities
>>> wrote ./out/manifest.json (5 global identities across 3 videos)
```

Output layout:

```text
out/
  manifest.json              # cross-video re-id result
  per_video/
    ep01/
      manifest.json          # per-video identities + tracks
      ep01_track01_frame0023_cluster1.png   # bbox crop, representative frame
      ...
    ep02/ ...
    ep03/ ...
```

The global manifest shape:

```json
{
  "videos": ["clips/ep01.mp4", "clips/ep02.mp4", "clips/ep03.mp4"],
  "per_video_summary": [
    { "video": "clips/ep01.mp4", "frames": 4320, "faces": 1237, "local_identities": 4 }
  ],
  "global_identities": [
    {
      "cluster_id": 1,
      "appearances": [
        { "video": "clips/ep01.mp4", "track_id": 1, "first_ts_ms": 1320, "last_ts_ms": 24600, "best_score": 0.91 },
        { "video": "clips/ep03.mp4", "track_id": 2, "first_ts_ms": 4100, "last_ts_ms": 31200, "best_score": 0.88 }
      ]
    }
  ]
}
```

## Tuning knobs

`VideoIdConfig` (`src/video_id.rs:31`):

| Field | Default | Effect |
|---|---|---|
| `track_iou` | `0.3` | IoU threshold for a detection to attach to an existing track. Lower = forgiving of fast motion; higher = stricter face-to-face association. |
| `track_max_age` | `8` | Frames a track survives without a match before retiring. ~8 frames at 24 fps absorbs a normal cut. |
| `cluster_threshold` | `0.36` | Cosine similarity above which two track mean embeddings merge. Mirrors `embedding::MatchConfig::threshold` — calibrate both together. |
| `min_embeddings_per_track` | `2` | Tracks below this many embeddings are recorded as loners, not clusters. Prevents a single noisy detection from claiming a new identity. |
| `min_face_px` | `32` | Smaller faces are skipped at the tracker level. Sub-32px crops produce garbage embeddings that would just add noise. |

## Plugging in industrial accuracy

`Identify` is a trait in `src/video_id.rs`. The closure adapter calls a
(detector, recogniser) pair; an override lets you fuse them (e.g. share
one ONNX session for both). For SCRFD + ArcFace the glue is now trivial:
`embed_all` already returns the sparse, index-tagged pairs the contract
requires.

```rust,ignore
use rsface::video_id::Identify;
use rsface::arcface_recognizer::ArcFaceRecognizer;
use rsface::scrfd_detector::ScrfdDetector;

struct Pipeline { det: ScrfdDetector, rec: ArcFaceRecognizer }

impl Identify for Pipeline {
    type Err = OnnxError;
    fn detect_and_embed(
        &mut self,
        _gray: &GrayImage,
        rgb: Option<&RgbImage>,
    ) -> Result<(Vec<Detection>, Vec<(usize, Embedding)>), Self::Err> {
        let rgb = rgb.expect("ArcFace needs RGB");
        let dets = self.det.detect_rgb(rgb)?;
        // (detection index, embedding); detections without landmarks are
        // simply absent, and the tracker drops those detections.
        let pairs = self.rec.embed_all(rgb, &dets)?;
        Ok((dets, pairs))
    }
}
```

This is the only piece of glue that needs to change to swap the detector /
recogniser. Everything downstream — tracker, clusterer, manifest writer —
is backend-agnostic.

## When to NOT use this

- **You already have a labelled gallery.** Use `Gallery::identify` directly
  (`src/embedding.rs`). `video_id` is for "no labelled gallery exists yet,
  group the unknown faces into identities."
- **You need frame-accurate bounding-box tracking.** `LightTracker` is
  IoU-only; it doesn't model motion. For Kalman / SORT use `sort-rs` or a
  similar crate and feed its `track_id` into `video_id` as the source
  `track_id` (skip the IoU step).
- **You need face quality / pose-aware re-id.** Cosine-on-512-d is the
  cheap baseline; on a real short-drama corpus with extreme pose / occlu-
  sion it underperforms a dedicated re-id model trained with hard mining.
  See `docs/recognition-lbph.md` for the accuracy-on-real-faces number —
  the video-level pipeline inherits the same ceiling.
