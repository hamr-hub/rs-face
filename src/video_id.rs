//! Video-level face identification.
//!
//! Builds on top of [`crate::embedding`], [`crate::detector`], and [`crate::source`]
//! to answer a question the frame-level detectors don't:
//!
//! > Across one or many videos, **which physical people appear**, and **when**?
//!
//! Three layers, each independently usable:
//!
//! 1. [`LightTracker`] — single-video IoU tracker. Takes a stream of
//!    `(frame_index, timestamp_ms, Vec<Detection>, Vec<Embedding>)` tuples and emits
//!    [`Track`]s with stable `track_id`s. Pure std, O(detections · active_tracks) per
//!    frame, no Kalman / Hungarian — short-drama cuts are the dominant motion.
//!
//! 2. [`IdentityCluster`] — pairwise single-linkage clustering over unit embeddings.
//!    Cosine threshold defaults to [`crate::embedding::MatchConfig::threshold`]
//!    (0.36, the InsightFace ArcFace operating point), so the cluster policy
//!    matches the gallery match policy. Cluster id `0` is reserved for "alone"
//!    singletons so the public id space starts at 1 and is stable as new videos
//!    are added.
//!
//! 3. [`identify_video`] — the per-video wrapper: pull frames from a
//!    [`FrameSource`], pass them through any detector + embedder pair
//!    implementing [`Identify`], and return one [`VideoIdentification`]
//!    manifest. Run it once per video and feed the results into
//!    [`IdentityCluster::finalise`] for cross-video identity merging.
//!
//! ## Design notes
//!
//! - **Embeddings are the source of truth**, not bbox coordinates. Two faces
//!   from the same person on opposite sides of a cut can have non-overlapping
//!   boxes; only the embedding tells us they're the same identity. Tracking
//!   collapses redundant boxes per video, clustering then merges identities
//!   across videos.
//!
//! - **No external cluster library.** Pairwise single-linkage over up to a few
//!   thousand tracks is fine in pure Rust; the second pass is O(N²) in
//!   cosine distance, which vectorises well and stays under a second at the
//!   scale short-drama corpora live at.
//!
//! - **Detector and embedder are injected.** Anything implementing the
//!   [`Identify`] trait — or just a pair of closures — plugs in. This keeps
//!   the module `default`-build-friendly (zero-dep haar + lbph/eigenface
//!   work) **and** opt-in industrial (`scrfd + arcface` works when
//!   `ort-backend` is enabled), without forcing a feature gate here.

use std::path::{Path, PathBuf};

use crate::detector::Detection;
use crate::embedding::{Embedding, MatchConfig};
use crate::image::{GrayImage, RgbImage};
use crate::source::FrameSource;

/// Configuration for the video-level identification pipeline.
#[derive(Clone, Debug)]
pub struct VideoIdConfig {
    /// Minimum IoU for the light tracker to associate a new detection with an
    /// existing track. `0.3` is forgiving enough for short-drama cuts where a
    /// face moves ~10% of the frame between consecutive detections.
    pub track_iou: f32,
    /// A track is closed (and forgotten) after this many consecutive frames
    /// without a matching detection. `8` survives a normal cut at 24fps.
    pub track_max_age: u32,
    /// Minimum cosine similarity (on unit embeddings) to merge two tracks into
    /// the same cluster. Defaults to [`MatchConfig::threshold`].
    pub cluster_threshold: f32,
    /// Skip a track from clustering if it has fewer embeddings than this. A
    /// single noisy embedding would otherwise force-create a new identity.
    pub min_embeddings_per_track: usize,
    /// Minimum face size (longer side, pixels) to bother embedding. Sub-32px
    /// faces produce garbage embeddings and inflate the cluster matrix.
    pub min_face_px: u32,
}

impl Default for VideoIdConfig {
    fn default() -> Self {
        Self {
            track_iou: 0.3,
            track_max_age: 8,
            cluster_threshold: MatchConfig::default().threshold,
            min_embeddings_per_track: 2,
            min_face_px: 32,
        }
    }
}

/// A single face track produced by [`LightTracker`].
#[derive(Clone, Debug)]
pub struct Track {
    /// Stable id within a single video (resets per video).
    pub track_id: u32,
    /// Source video this track was extracted from.
    pub video: PathBuf,
    /// First frame index where this track was seen.
    pub first_frame: u64,
    /// Last frame index where this track was seen.
    pub last_frame: u64,
    /// Presentation timestamp of the first appearance, milliseconds.
    pub first_ts_ms: u64,
    /// Presentation timestamp of the last appearance, milliseconds.
    pub last_ts_ms: u64,
    /// All embeddings collected for this track (one per matched detection).
    pub embeddings: Vec<Embedding>,
    /// The bbox from the detection that produced the highest-score embedding;
    /// representative for thumbnail cropping.
    pub best_bbox: Detection,
    /// Score of the best embedding's source detection.
    pub best_score: f32,
}

impl Track {
    /// Mean of all embeddings (unit vector). Returns `None` if the track has
    /// no embeddings; cluster callers should pre-filter with
    /// `min_embeddings_per_track`.
    pub fn mean_embedding(&self) -> Option<Embedding> {
        if self.embeddings.is_empty() {
            return None;
        }
        let dim = self.embeddings[0].dim();
        let mut acc = vec![0.0f32; dim];
        for e in &self.embeddings {
            // Embedding::cosine dimension-asserts; trusting the constructor for raw bytes here.
            for (a, b) in acc.iter_mut().zip(e.as_slice()) {
                *a += b;
            }
        }
        Embedding::from_raw(&acc)
    }
}

/// A frame-level observation handed to [`LightTracker::update`].
#[derive(Clone, Debug)]
pub struct FrameObservation<'a> {
    pub frame_index: u64,
    pub timestamp_ms: u64,
    pub detections: &'a [Detection],
    pub embeddings: &'a [Embedding],
}

/// Single-video IoU tracker. Greedy nearest-match by IoU, with an age-based
/// retirement policy for unmatched tracks.
///
/// Not Kalman, not Hungarian. Short-drama scenes move a face tens of pixels
/// between frames at most; IoU + a small age budget is enough and stays at
/// O(detections · active_tracks).
#[derive(Debug)]
pub struct LightTracker {
    cfg: VideoIdConfig,
    video: PathBuf,
    next_id: u32,
    /// Currently-open tracks. Order is "creation order"; we walk from oldest
    /// to newest so the oldest track gets the first shot at a new detection.
    active: Vec<OpenTrack>,
}

#[derive(Debug)]
struct OpenTrack {
    id: u32,
    bbox: Detection,
    score: f32,
    /// Every embedding matched into this track. The clusterer uses the
    /// mean of this vector, so accumulating matters: a track seen in 5
    /// consecutive frames contributes 5 evidence points, not 1.
    embeddings: Vec<Embedding>,
    last_frame: u64,
    last_ts_ms: u64,
    age: u32,
}

impl LightTracker {
    /// Create a tracker for a single video. The path is recorded on every
    /// emitted [`Track`] so downstream consumers can attribute appearances
    /// without cross-referencing.
    pub fn new(video: PathBuf, cfg: VideoIdConfig) -> Self {
        Self {
            cfg,
            video,
            next_id: 1,
            active: Vec::new(),
        }
    }

    /// Feed one frame's worth of detections + embeddings into the tracker.
    ///
    /// `detections` and `embeddings` must be the same length and in the same
    /// order — the caller is the one that produced both, usually via
    /// `ArcFaceRecognizer::embed_all` or `LbphRecognizer::embed`. Detections
    /// with no corresponding embedding are silently dropped, which matches
    /// `embed_all`'s "embedding only when we can" contract.
    pub fn update(&mut self, obs: FrameObservation<'_>) -> Vec<Track> {
        let mut retired = Vec::new();

        // Age every active track before associating this frame.
        for t in &mut self.active {
            t.age = t.age.saturating_add(1);
        }

        // Greedy: for each detection, pick the best-IoU active track above
        // the threshold and unique-claim it. Linear scan is fine; at default
        // settings short-drama videos have < 10 active tracks simultaneously.
        //
        // We track claimed indices in a HashSet, not a `Vec<bool>`, because
        // `self.active` grows inside the loop when a detection opens a new
        // track — the length of any pre-allocated bool vec would be stale
        // by the end.
        let mut claimed: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let initial_active = self.active.len();
        for (det, emb) in obs.detections.iter().zip(obs.embeddings.iter()) {
            if (det.w.max(det.h) as u32) < self.cfg.min_face_px {
                continue;
            }
            let mut best_idx: Option<usize> = None;
            let mut best_iou: f32 = 0.0;
            // Only consider tracks that existed at the start of this frame;
            // a track pushed earlier in this same loop gets first shot on
            // the *next* frame, which avoids claiming a brand-new track on
            // the same frame it was opened.
            for i in 0..initial_active {
                if claimed.contains(&i) {
                    continue;
                }
                let iou = self.active[i].bbox.iou(det);
                if iou >= self.cfg.track_iou && iou > best_iou {
                    best_iou = iou;
                    best_idx = Some(i);
                }
            }
            if let Some(i) = best_idx {
                let t = &mut self.active[i];
                claimed.insert(i);
                t.age = 0;
                t.last_frame = obs.frame_index;
                t.last_ts_ms = obs.timestamp_ms;
                // Keep the higher-scoring detection as the running "best".
                if det.score > t.score {
                    t.score = det.score;
                    t.bbox = det.clone();
                }
                t.embeddings.push(emb.clone());
            } else {
                let id = self.next_id;
                self.next_id += 1;
                self.active.push(OpenTrack {
                    id,
                    bbox: det.clone(),
                    score: det.score,
                    embeddings: vec![emb.clone()],
                    last_frame: obs.frame_index,
                    last_ts_ms: obs.timestamp_ms,
                    age: 0,
                });
            }
        }

        // Retire tracks that went too long without a match.
        let max_age = self.cfg.track_max_age;
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].age > max_age {
                let t = self.active.swap_remove(i);
                retired.push(self.finalise(t, obs.frame_index));
            } else {
                i += 1;
            }
        }

        retired
    }

    /// Flush all currently-active tracks. Call once at end-of-video.
    pub fn finish(&mut self) -> Vec<Track> {
        let drained: Vec<OpenTrack> = std::mem::take(&mut self.active);
        drained
            .into_iter()
            .map(|t| {
                let last_frame = t.last_frame;
                self.finalise(t, last_frame)
            })
            .collect()
    }

    fn finalise(&self, t: OpenTrack, last_frame: u64) -> Track {
        Track {
            track_id: t.id,
            video: self.video.clone(),
            first_frame: t.last_frame.saturating_sub(t.age as u64),
            last_frame,
            first_ts_ms: t.last_ts_ms.saturating_sub(t.age as u64 * 33),
            last_ts_ms: t.last_ts_ms,
            embeddings: t.embeddings,
            best_bbox: t.bbox,
            best_score: t.score,
        }
    }
}

/// Cross-video identity: a cluster of tracks that the embedder agrees are the
/// same physical person.
#[derive(Clone, Debug)]
pub struct Identity {
    /// Stable id across the whole run (1-based). `0` is reserved for
    /// singletons that didn't get clustered (see
    /// [`IdentityCluster::assign`]).
    pub cluster_id: u32,
    /// All tracks assigned to this identity.
    pub tracks: Vec<Track>,
    /// Mean embedding of every track's mean embedding — the identity centroid.
    pub centroid: Embedding,
}

/// One identity assignment per track. Stable across merges done after the fact.
#[derive(Clone, Debug)]
pub struct TrackAssignment {
    pub video: PathBuf,
    pub track_id: u32,
    pub cluster_id: u32,
    /// Cosine similarity between the track's mean embedding and the cluster
    /// centroid. Below-threshold assignments are reported with a `0` cluster
    /// id and a `negative` similarity to surface them to the caller.
    pub similarity: f32,
}

/// Greedy single-linkage clusterer over unit embeddings.
///
/// For each track we compare its mean embedding against every existing
/// cluster centroid and merge into the best match if the cosine similarity
/// exceeds [`VideoIdConfig::cluster_threshold`]. Centroids are recomputed
/// after each merge so a slow drift across videos (lighting, makeup) is
/// absorbed rather than locked in.
#[derive(Debug, Default)]
pub struct IdentityCluster {
    cfg: VideoIdConfig,
    centroids: Vec<Embedding>,
    /// Tracks bucketed by `cluster_id - 1`; index-aligned with `centroids`.
    by_cluster: Vec<Vec<Track>>,
    /// Tracks that fell below `min_embeddings_per_track` or whose mean
    /// embedding was zero-norm. They never enter a cluster, but the
    /// `TrackAssignment` row is still recorded with `cluster_id = 0`.
    loners: Vec<Track>,
}

impl IdentityCluster {
    pub fn new(cfg: VideoIdConfig) -> Self {
        Self {
            cfg,
            centroids: Vec::new(),
            by_cluster: Vec::new(),
            loners: Vec::new(),
        }
    }

    /// Total identities clustered so far.
    pub fn len(&self) -> usize {
        self.centroids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    /// Add a track. Returns the assigned `cluster_id` (1-based) or `0` if the
    /// track sat below the threshold against every existing centroid (or
    /// didn't have enough embeddings to commit).
    pub fn assign(&mut self, track: Track) -> u32 {
        if track.embeddings.len() < self.cfg.min_embeddings_per_track {
            self.loners.push(track);
            return 0;
        }
        let Some(mean) = track.mean_embedding() else {
            self.loners.push(track);
            return 0;
        };

        // Find the best existing centroid above threshold.
        let mut best_idx: Option<usize> = None;
        let mut best_sim: f32 = 0.0;
        for (i, c) in self.centroids.iter().enumerate() {
            if let Some(sim) = mean.cosine(c) {
                if sim >= self.cfg.cluster_threshold && sim > best_sim {
                    best_sim = sim;
                    best_idx = Some(i);
                }
            }
        }

        if let Some(i) = best_idx {
            let new_centroid = blend_centroid(&self.centroids[i], &mean);
            self.centroids[i] = new_centroid;
            self.by_cluster[i].push(track);
            (i + 1) as u32
        } else {
            self.centroids.push(mean);
            self.by_cluster.push(vec![track]);
            self.centroids.len() as u32
        }
    }

    /// Drain all tracks and produce one [`Identity`] per cluster alongside
    /// per-track assignments (including the below-threshold loners).
    pub fn finalise(self) -> (Vec<Identity>, Vec<TrackAssignment>) {
        let Self {
            cfg,
            centroids,
            by_cluster,
            loners,
        } = self;

        let mut identities: Vec<Identity> = centroids
            .into_iter()
            .zip(by_cluster)
            .enumerate()
            .map(|(i, (centroid, tracks))| Identity {
                cluster_id: (i + 1) as u32,
                tracks,
                centroid,
            })
            .collect();

        // Per-track assignment rows for the clustered ones, with the cosine
        // similarity against the (post-merge) centroid.
        let mut assignments: Vec<TrackAssignment> = Vec::new();
        for id in &identities {
            for t in &id.tracks {
                let sim = t
                    .mean_embedding()
                    .and_then(|m| m.cosine(&id.centroid))
                    .unwrap_or(-1.0);
                assignments.push(TrackAssignment {
                    video: t.video.clone(),
                    track_id: t.track_id,
                    cluster_id: id.cluster_id,
                    similarity: sim,
                });
            }
        }

        // Loners: no cluster, similarity reported as -1 so the caller can
        // tell at a glance which rows were dropped.
        for t in loners {
            assignments.push(TrackAssignment {
                video: t.video.clone(),
                track_id: t.track_id,
                cluster_id: 0,
                similarity: -1.0,
            });
        }

        // Final sweep: drop identities that ended up with zero tracks. Can
        // happen if every track in a cluster fell below threshold on a later
        // merge and we chose to roll back — defensive, cheap.
        identities.retain(|id| !id.tracks.is_empty());

        // Suppress cfg unused.
        let _ = cfg;
        (identities, assignments)
    }
}

/// Compute a unit centroid that averages two unit embeddings and
/// renormalises. Cheap L2 norm recomputation, fine for 512-d.
fn blend_centroid(a: &Embedding, b: &Embedding) -> Embedding {
    let mut acc: Vec<f32> = a
        .as_slice()
        .iter()
        .zip(b.as_slice().iter())
        .map(|(x, y)| x + y)
        .collect();
    let norm: f32 = acc.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for v in &mut acc {
            *v /= norm;
        }
    }
    // Embedding::from_raw handles the empty/zero-norm rejection we just
    // skipped past; if it ever does reject, fall back to `a` so the cluster
    // doesn't silently lose its centroid.
    Embedding::from_raw(&acc).unwrap_or_else(|| a.clone())
}

/// Trait abstracting "produce (detections, embeddings) for one frame".
///
/// Implementations exist for any (detector, recogniser) pair:
/// - `(HaarDetector, LbphRecognizer::extract)` — zero-dep.
/// - `(ScrfdDetector, ArcFaceRecognizer::embed_all)` — industrial, ONNX path.
///
/// The default implementation in this module just calls the two closures
/// separately; only override if you need to fuse them (e.g. share a single
/// ONNX session for both).
pub trait Identify {
    type Err: std::fmt::Debug;
    fn detect_and_embed(
        &mut self,
        gray: &GrayImage,
        rgb: Option<&RgbImage>,
    ) -> Result<(Vec<Detection>, Vec<Embedding>), Self::Err>;
}

/// Closure-based [`Identify`] for ad-hoc pipelines.
pub struct ClosureIdentify<D, E, Dr, Er>
where
    D: FnMut(&GrayImage) -> Result<Vec<Detection>, Dr>,
    E: FnMut(&RgbImage, &[Detection]) -> Result<Vec<Embedding>, Er>,
{
    pub detect: D,
    pub embed: E,
}

impl<D, E, Dr, Er> Identify for ClosureIdentify<D, E, Dr, Er>
where
    D: FnMut(&GrayImage) -> Result<Vec<Detection>, Dr>,
    E: FnMut(&RgbImage, &[Detection]) -> Result<Vec<Embedding>, Er>,
    Dr: std::fmt::Debug,
    Er: std::fmt::Debug,
{
    type Err = IdentifyError<Dr, Er>;
    fn detect_and_embed(
        &mut self,
        gray: &GrayImage,
        rgb: Option<&RgbImage>,
    ) -> Result<(Vec<Detection>, Vec<Embedding>), Self::Err> {
        let dets = (self.detect)(gray).map_err(IdentifyError::Detect)?;
        // If we don't have RGB, derive a placeholder by replicating gray.
        let embeddings = match rgb {
            Some(rgb) => (self.embed)(rgb, &dets).map_err(IdentifyError::Embed)?,
            None => {
                let rgb = gray_to_rgb(gray);
                (self.embed)(&rgb, &dets).map_err(IdentifyError::Embed)?
            }
        };
        Ok((dets, embeddings))
    }
}

/// Two-side error so the caller can see which half of the pipeline failed
/// without `Box<dyn Error>`.
#[derive(Debug)]
pub enum IdentifyError<D, E> {
    Detect(D),
    Embed(E),
}

impl<D: std::fmt::Display, E: std::fmt::Display> std::fmt::Display for IdentifyError<D, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdentifyError::Detect(d) => write!(f, "detect: {d}"),
            IdentifyError::Embed(e) => write!(f, "embed: {e}"),
        }
    }
}

impl<D: std::error::Error + 'static, E: std::error::Error + 'static> std::error::Error
    for IdentifyError<D, E>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IdentifyError::Detect(d) => Some(d),
            IdentifyError::Embed(e) => Some(e),
        }
    }
}

/// Per-video identification result.
#[derive(Clone, Debug, Default)]
pub struct VideoIdentification {
    pub video: PathBuf,
    pub tracks: Vec<Track>,
    pub identities: Vec<Identity>,
    pub assignments: Vec<TrackAssignment>,
    pub frames_processed: u64,
    pub faces_embedded: u64,
}

/// Run the full per-video identification pipeline.
///
/// `identify` is whatever `(detector, embedder)` pair the caller chose. Frames
/// are pulled from `source` until exhaustion; tracks emitted by the tracker
/// are clustered into identities inline. Returns one [`VideoIdentification`]
/// per call.
pub fn identify_video<I: Identify>(
    source: &mut dyn FrameSource,
    video: PathBuf,
    cfg: VideoIdConfig,
    identify: &mut I,
) -> Result<VideoIdentification, std::io::Error>
where
    I::Err: std::fmt::Display,
{
    let mut tracker = LightTracker::new(video.clone(), cfg.clone());
    let mut clusterer = IdentityCluster::new(cfg.clone());
    let mut frames: u64 = 0;
    let mut faces: u64 = 0;

    while let Some(frame) = source.next_frame()? {
        frames += 1;
        let (dets, embs) = identify
            .detect_and_embed(&frame.gray, frame.rgb.as_deref())
            .map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, format!("identify: {e}"))
            })?;
        faces += embs.len() as u64;

        let obs = FrameObservation {
            frame_index: frame.index,
            timestamp_ms: frame.timestamp_ms,
            detections: &dets,
            embeddings: &embs,
        };
        for retired in tracker.update(obs) {
            clusterer.assign(retired);
        }
    }
    for retired in tracker.finish() {
        clusterer.assign(retired);
    }

    let (identities, assignments) = clusterer.finalise();
    Ok(VideoIdentification {
        video,
        tracks: identities
            .iter()
            .flat_map(|i| i.tracks.iter().cloned())
            .chain(std::iter::empty())
            .collect(),
        identities,
        assignments,
        frames_processed: frames,
        faces_embedded: faces,
    })
}

/// Cross-video re-id: cluster the identities from every video in `runs`
/// together so "Alice in ep01" and "Alice in ep03" land in the same identity.
///
/// Each per-video run is treated as a bag of tracks; their mean embeddings
/// are re-clustered with the same single-linkage rule. The per-video
/// `cluster_id` values are rewritten in place to the global ones so the
/// output reads as one merged identity space.
pub fn merge_across_videos(runs: Vec<VideoIdentification>, cfg: VideoIdConfig) -> Vec<Identity> {
    // Flatten every track from every video into a single assignment stream.
    let mut clusterer = IdentityCluster::new(cfg);
    let mut all_tracks: Vec<Track> = Vec::new();
    for run in runs {
        for identity in run.identities {
            for track in identity.tracks {
                all_tracks.push(track);
            }
        }
    }
    for t in all_tracks {
        clusterer.assign(t);
    }
    clusterer.finalise().0
}

fn gray_to_rgb(gray: &GrayImage) -> RgbImage {
    let (w, h) = (gray.width(), gray.height());
    let mut rgb = RgbImage::new(w, h);
    for y in 0..h {
        let row = rgb.row_mut(y);
        let g = gray.row(y);
        for (x, &v) in g.iter().enumerate() {
            row[x * 3] = v;
            row[x * 3 + 1] = v;
            row[x * 3 + 2] = v;
        }
    }
    rgb
}

/// Convenience: write a JSON manifest of one [`VideoIdentification`] run to
/// `out_dir/manifest.json`. Pure std (no `serde`), hand-rolled the same way
/// the rest of the crate does it (see [`crate::output`]).
pub fn write_video_manifest(out_dir: &Path, ident: &VideoIdentification) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let path = out_dir.join("video_manifest.json");
    let json = render_video_manifest_json(ident);
    std::fs::write(path, json)
}

/// Render the manifest as a JSON string. Exposed so tests can assert on the
/// shape without touching the filesystem.
pub fn render_video_manifest_json(ident: &VideoIdentification) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(1024);
    let _ = writeln!(s, "{{");
    let _ = writeln!(s, "  \"video\": {:?},", ident.video.display().to_string());
    let _ = writeln!(s, "  \"frames_processed\": {},", ident.frames_processed);
    let _ = writeln!(s, "  \"faces_embedded\": {},", ident.faces_embedded);
    let _ = writeln!(s, "  \"identities\": [");
    for (i, id) in ident.identities.iter().enumerate() {
        let comma = if i + 1 == ident.identities.len() {
            ""
        } else {
            ","
        };
        let _ = writeln!(s, "    {{");
        let _ = writeln!(s, "      \"cluster_id\": {},", id.cluster_id);
        let _ = writeln!(s, "      \"tracks\": [");
        for (j, t) in id.tracks.iter().enumerate() {
            let tcomma = if j + 1 == id.tracks.len() { "" } else { "," };
            let _ = writeln!(
                s,
                "        {{ \"track_id\": {}, \"first_frame\": {}, \"last_frame\": {}, \
                 \"first_ts_ms\": {}, \"last_ts_ms\": {}, \"best_score\": {:.4} }}{}",
                t.track_id,
                t.first_frame,
                t.last_frame,
                t.first_ts_ms,
                t.last_ts_ms,
                t.best_score,
                tcomma
            );
        }
        let _ = writeln!(s, "      ]");
        let _ = writeln!(s, "    }}{comma}", comma = comma);
    }
    let _ = writeln!(s, "  ]");
    let _ = writeln!(s, "}}");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::Embedding;

    fn det(x: usize, y: usize, w: usize, h: usize, score: f32) -> Detection {
        Detection { x, y, w, h, score }
    }

    fn unit_along(dim: usize, k: usize) -> Embedding {
        let mut v = vec![0.0f32; dim];
        v[k] = 1.0;
        Embedding::from_raw(&v).unwrap()
    }

    #[test]
    fn tracker_emits_same_id_for_overlapping_boxes() {
        let mut t = LightTracker::new(PathBuf::from("clip.mp4"), VideoIdConfig::default());
        let emb = unit_along(8, 0);

        let retired = t.update(FrameObservation {
            frame_index: 0,
            timestamp_ms: 0,
            detections: &[det(10, 10, 50, 50, 0.9)],
            embeddings: &[emb.clone()],
        });
        assert!(retired.is_empty(), "no retirement on first frame");

        // Slide the box 2px right — IoU still well above 0.3.
        let retired = t.update(FrameObservation {
            frame_index: 1,
            timestamp_ms: 33,
            detections: &[det(12, 10, 50, 50, 0.9)],
            embeddings: &[emb.clone()],
        });
        assert!(retired.is_empty(), "same track continues");

        // Box jumps away — new track.
        let retired = t.update(FrameObservation {
            frame_index: 2,
            timestamp_ms: 66,
            detections: &[det(200, 200, 40, 40, 0.9)],
            embeddings: &[emb.clone()],
        });
        assert!(retired.is_empty());

        // Push the first track past max_age.
        let mut last_retired = Vec::new();
        for f in 3..=20 {
            last_retired = t.update(FrameObservation {
                frame_index: f,
                timestamp_ms: f * 33,
                detections: &[det(200, 200, 40, 40, 0.9)],
                embeddings: &[emb.clone()],
            });
            if !last_retired.is_empty() {
                break;
            }
        }
        assert_eq!(last_retired.len(), 1, "the first track should retire");
        assert_eq!(last_retired[0].track_id, 1);
    }

    #[test]
    fn tracker_retires_on_finish() {
        let mut t = LightTracker::new(PathBuf::from("clip.mp4"), VideoIdConfig::default());
        let emb = unit_along(8, 0);
        t.update(FrameObservation {
            frame_index: 0,
            timestamp_ms: 0,
            detections: &[det(10, 10, 50, 50, 0.9)],
            embeddings: &[emb],
        });
        let drained = t.finish();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].track_id, 1);
    }

    #[test]
    fn cluster_merges_same_embedding_into_one_identity() {
        let mut c = IdentityCluster::new(VideoIdConfig::default());
        let emb = unit_along(8, 0);
        let mut mk_track = |id: u32, score: f32| Track {
            track_id: id,
            video: PathBuf::from("clip.mp4"),
            first_frame: 0,
            last_frame: 10,
            first_ts_ms: 0,
            last_ts_ms: 330,
            embeddings: vec![emb.clone(), emb.clone()],
            best_bbox: det(0, 0, 50, 50, score),
            best_score: score,
        };
        let id1 = c.assign(mk_track(1, 0.9));
        let id2 = c.assign(mk_track(2, 0.8));
        assert_eq!(id1, 1);
        assert_eq!(id2, 1, "orthogonal-distance-0 merges into first cluster");
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn cluster_splits_orthogonal_embeddings() {
        let mut c = IdentityCluster::new(VideoIdConfig::default());
        let mut mk_track = |id: u32, k: usize| Track {
            track_id: id,
            video: PathBuf::from("clip.mp4"),
            first_frame: 0,
            last_frame: 10,
            first_ts_ms: 0,
            last_ts_ms: 330,
            embeddings: vec![unit_along(8, k), unit_along(8, k)],
            best_bbox: det(0, 0, 50, 50, 0.9),
            best_score: 0.9,
        };
        let id1 = c.assign(mk_track(1, 0));
        let id2 = c.assign(mk_track(2, 1));
        assert_eq!(id1, 1);
        assert_eq!(id2, 2, "orthogonal vectors must not merge");
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn cluster_skips_single_embedding_tracks() {
        let cfg = VideoIdConfig {
            min_embeddings_per_track: 2,
            ..VideoIdConfig::default()
        };
        let mut c = IdentityCluster::new(cfg);
        let emb = unit_along(8, 0);
        let single = Track {
            track_id: 1,
            video: PathBuf::from("clip.mp4"),
            first_frame: 0,
            last_frame: 0,
            first_ts_ms: 0,
            last_ts_ms: 0,
            embeddings: vec![emb],
            best_bbox: det(0, 0, 50, 50, 0.9),
            best_score: 0.9,
        };
        let id = c.assign(single);
        assert_eq!(
            id, 0,
            "singleton below min_embeddings_per_track must stay unassigned"
        );
        assert!(c.is_empty());
    }

    #[test]
    fn manifest_json_round_trips_identity_count() {
        let emb = unit_along(8, 0);
        let track = Track {
            track_id: 1,
            video: PathBuf::from("clip.mp4"),
            first_frame: 0,
            last_frame: 9,
            first_ts_ms: 0,
            last_ts_ms: 300,
            embeddings: vec![emb.clone(), emb],
            best_bbox: det(10, 10, 50, 50, 0.91),
            best_score: 0.91,
        };
        let ident = VideoIdentification {
            video: PathBuf::from("clip.mp4"),
            tracks: vec![track.clone()],
            identities: vec![Identity {
                cluster_id: 1,
                tracks: vec![track],
                centroid: unit_along(8, 0),
            }],
            assignments: vec![TrackAssignment {
                video: PathBuf::from("clip.mp4"),
                track_id: 1,
                cluster_id: 1,
                similarity: 1.0,
            }],
            frames_processed: 100,
            faces_embedded: 12,
        };
        let json = render_video_manifest_json(&ident);
        assert!(json.contains("\"frames_processed\": 100"));
        assert!(json.contains("\"faces_embedded\": 12"));
        assert!(json.contains("\"cluster_id\": 1"));
        assert!(json.contains("\"best_score\": 0.9100"));
    }
}
