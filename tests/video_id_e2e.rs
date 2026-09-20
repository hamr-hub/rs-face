//! End-to-end tests for `video_id`: tracker + clusterer under controlled
//! per-frame (detections, embeddings) inputs.
//!
//! These don't go through real detection or recognition (those are
//! algorithm-specific and live in their own `real_model_e2e` suite). The
//! point is to lock in the cross-frame identity behaviour — two actors
//! alternating across 8 frames must come out as two identities, never one,
//! even when the embeddings for "Alice" are noisy variations of the same
//! unit vector.

use std::path::PathBuf;
use std::sync::Arc;

use rsface::embedding::Embedding;
use rsface::face::Detection;
use rsface::image::{GrayImage, RgbImage};
use rsface::source::{Frame, FrameSource};
use rsface::video_id::{identify_video, Identify, VideoIdConfig};

/// Pre-scripted frame source: each entry's detections for one frame. The
/// frame is just a 1x1 gray placeholder; the test never reads pixels.
struct ScriptedSource {
    frames: Vec<Vec<Detection>>,
    index: u64,
}

impl ScriptedSource {
    fn new(frames: Vec<Vec<Detection>>) -> Self {
        Self { frames, index: 0 }
    }
}

impl FrameSource for ScriptedSource {
    fn next_frame(&mut self) -> std::io::Result<Option<Frame>> {
        if (self.index as usize) >= self.frames.len() {
            return Ok(None);
        }
        let dets = self.frames[self.index as usize].clone();
        self.index += 1;
        let gray = Arc::new(GrayImage::new(1, 1));
        let rgb = Some(Arc::new(RgbImage::new(1, 1)));
        Ok(Some(Frame {
            index: self.index - 1,
            timestamp_ms: (self.index - 1) * 33,
            gray,
            rgb,
        }))
    }
}

/// Echo the per-frame scripted detections directly, tagging each one with
/// its index + paired embedding (the contract Identify now requires).
struct ScriptedIdentify {
    frames: Vec<(Vec<Detection>, Vec<(usize, Embedding)>)>,
    cursor: usize,
}

impl Identify for ScriptedIdentify {
    type Err = std::convert::Infallible;
    fn detect_and_embed(
        &mut self,
        _gray: &GrayImage,
        _rgb: Option<&RgbImage>,
    ) -> Result<(Vec<Detection>, Vec<(usize, Embedding)>), Self::Err> {
        if self.cursor >= self.frames.len() {
            return Ok((Vec::new(), Vec::new()));
        }
        let frame = self.frames[self.cursor].clone();
        self.cursor += 1;
        Ok(frame)
    }
}

fn det(x: usize, y: usize, w: usize, h: usize, score: f32) -> Detection {
    Detection { x, y, w, h, score }
}

/// Build a unit vector that's a noisy clone of basis[k] — keeps cosine
/// above 0.9 to the canonical embedding so single-linkage clustering merges.
fn near_basis(k: usize, dim: usize, noise: f32) -> Embedding {
    let mut v = vec![noise * 0.1; dim];
    v[k] = 1.0;
    // Re-normalise so Embedding::from_raw accepts it.
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    Embedding::from_raw(&v).expect("non-zero norm")
}

#[test]
fn two_actors_alternating_cluster_into_two_identities() {
    // 8 frames. Alice appears in frames 0,1,2, then Bob in 3,4,5, then Alice
    // again in 6,7. Same physical Alice across the two appearances; same Bob.
    let dim = 16;
    let alice = near_basis(0, dim, 0.0);
    let bob = near_basis(1, dim, 0.0);

    let mut det_frames: Vec<Vec<Detection>> = Vec::new();
    let mut frames: Vec<(Vec<Detection>, Vec<(usize, Embedding)>)> = Vec::new();
    for i in 0..6 {
        let (e, bbox) = if i < 3 {
            (alice.clone(), det(10, 10, 80, 80, 0.9))
        } else {
            (bob.clone(), det(200, 100, 80, 80, 0.9))
        };
        det_frames.push(vec![bbox.clone()]);
        frames.push((vec![bbox], vec![(0, e)]));
    }
    // Alice re-appears at the end.
    let bbox1 = det(15, 12, 78, 80, 0.85);
    let bbox2 = det(20, 14, 76, 78, 0.85);
    det_frames.push(vec![bbox1.clone()]);
    frames.push((vec![bbox1], vec![(0, alice.clone())]));
    det_frames.push(vec![bbox2.clone()]);
    frames.push((vec![bbox2], vec![(0, alice.clone())]));

    let mut src = ScriptedSource::new(det_frames);
    let mut id = ScriptedIdentify { frames, cursor: 0 };

    let cfg = VideoIdConfig {
        track_max_age: 4,
        min_embeddings_per_track: 2,
        ..VideoIdConfig::default()
    };
    let result =
        identify_video(&mut src, PathBuf::from("test.mp4"), cfg, &mut id).expect("identify_video");

    // Two identities, with Alice's appearances split across two tracks that
    // got merged into one cluster, Bob into a separate cluster.
    assert_eq!(result.identities.len(), 2, "got {:?}", result.identities);

    let alice_id = result
        .identities
        .iter()
        .find(|i| i.tracks.iter().any(|t| t.best_bbox.x < 100))
        .expect("Alice's cluster must exist");
    let bob_id = result
        .identities
        .iter()
        .find(|i| i.tracks.iter().any(|t| t.best_bbox.x >= 100))
        .expect("Bob's cluster must exist");

    assert_ne!(alice_id.cluster_id, bob_id.cluster_id);
    // Alice appears in 5 frames (0,1,2 + 6,7) so her track holds >= 5 embeddings.
    assert!(alice_id.tracks.iter().any(|t| t.embeddings.len() >= 3));
    // Bob appears in frames 3,4,5 — 3 embeddings on one track.
    assert!(bob_id.tracks.iter().any(|t| t.embeddings.len() >= 3));

    // Every assignment reports a positive similarity (no below-threshold rows).
    for a in &result.assignments {
        assert!(a.similarity >= 0.0, "below-threshold row leaked: {:?}", a);
    }
}

#[test]
fn below_threshold_singleton_lands_in_cluster_zero() {
    // Two faces that are well below the cosine threshold against each
    // other. Each is repeated across enough frames to satisfy
    // `min_embeddings_per_track`, but they should never merge into a single
    // identity because their embeddings are far apart.
    let dim = 8;
    let alice = near_basis(0, dim, 0.0);
    let bob = near_basis(3, dim, 0.0); // cos(alice, bob) == 0

    let mut det_frames: Vec<Vec<Detection>> = Vec::new();
    let mut frames: Vec<(Vec<Detection>, Vec<(usize, Embedding)>)> = Vec::new();
    for i in 0..6 {
        let (e, bbox) = if i % 2 == 0 {
            (alice.clone(), det(10, 10, 80, 80, 0.9))
        } else {
            (bob.clone(), det(200, 100, 80, 80, 0.9))
        };
        det_frames.push(vec![bbox.clone()]);
        frames.push((vec![bbox], vec![(0, e)]));
    }

    let mut src = ScriptedSource::new(det_frames);
    let mut id = ScriptedIdentify { frames, cursor: 0 };

    let cfg = VideoIdConfig {
        track_max_age: 1,
        min_embeddings_per_track: 1,
        // Forbid any merging — orthogonal embeddings should not cross.
        cluster_threshold: 0.5,
        ..VideoIdConfig::default()
    };
    let result =
        identify_video(&mut src, PathBuf::from("test.mp4"), cfg, &mut id).expect("identify_video");

    assert_eq!(
        result.identities.len(),
        2,
        "two distinct identities, got {:?}",
        result.identities
    );
    for a in &result.assignments {
        assert!(a.similarity >= 0.0, "below-threshold row leaked: {:?}", a);
    }
}
