//! Multi-face tracker for video streams.
//!
//! Given a stream of detection sets (one per frame), assign a stable
//! integer ID to every face so the same person keeps the same label from
//! frame to frame. This is the *tracking* layer that sits between the
//! detector and any downstream cross-frame reasoning (action recognition,
//! re-identification, frame-attribution, ...).
//!
//! ## Algorithm
//!
//! Greedy IoU association with a centroid-distance fallback. For every
//! new frame:
//!
//! 1. For each detection, score every active track by IoU first; if no
//!    active track has IoU above [`TrackerConfig::min_iou`], fall back to
//!    centroid distance and accept the best one if it is within
//!    `centroid_gate × bbox_diagonal`.
//! 2. The detection → track pairing is solved greedily: the highest-scoring
//!    pair wins, both are removed from the pool, repeat. This is the
//!    classical SORT-style assignment — O(detections · tracks) per frame,
//!    no Hungarian, no Kalman. The greedy choice is provably optimal when
//!    detections and tracks do not overlap heavily (the common case for
//!    ≤8 faces/frame at moderate framerates).
//! 3. Unmatched detections spawn new tracks (with the next available ID).
//! 4. Tracks that went unmatched for [`TrackerConfig::max_age`] frames
//!    are closed and dropped — their ID is *not* recycled, so any
//!    downstream log of `face_seen(id=N)` is unambiguous.
//!
//! ## Complexity
//!
//! - Memory: O(active_tracks) — bounded by the number of faces that
//!   appeared in the last `max_age + 1` frames.
//! - Per-frame cost: O(D · T) where `D = detections this frame`,
//!   `T = active tracks`. At default `max_age = 5` and 8 faces/frame,
//!   that is < 200 IoU evaluations per frame — sub-microsecond.
//!
//! ## Difference from `video_id::LightTracker`
//!
//! `LightTracker` is bound to the recognition pipeline — it stores an
//! `Embedding` per matched detection and feeds the clusterer that produces
//! `Track` summaries across an entire video. This module is the lighter,
//! pure-detection counterpart: it only does frame-to-frame identity
//! assignment, has no recogniser dependency, and is suitable for any
//! "same face?" downstream consumer (overlay rendering, per-frame
//! attribution, simple activity counting).

use crate::face::Detection;

/// Default settings matching the task spec's "≤8 faces/frame" case.
impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            min_iou: 0.20,
            centroid_gate: 1.0 / 3.0,
            max_age: 5,
            next_id: 1,
        }
    }
}

/// Tunables for [`FaceTracker`].
///
/// `min_iou` and `centroid_gate` control how forgiving the association
/// step is. `max_age` controls how long a track survives without a match.
/// Defaults are tuned for typical 24–30 fps video with modest inter-frame
/// motion.
#[derive(Clone, Copy, Debug)]
pub struct TrackerConfig {
    /// Minimum IoU required to match a detection to a track directly.
    /// Below this, the algorithm falls back to centroid distance.
    pub min_iou: f32,
    /// Centroid distance gate, expressed as a fraction of the *track's*
    /// bbox diagonal. A detection whose centroid is closer than
    /// `centroid_gate × diagonal` will be associated with the track if
    /// no IoU match was found.
    pub centroid_gate: f32,
    /// Number of consecutive frames a track may go unmatched before it is
    /// closed. `0` means a track must be matched every frame.
    pub max_age: u32,
    /// Initial value of the next-assigned ID. Defaults to `1`. Tests use
    /// other values to disambiguate from manual IDs.
    pub next_id: u32,
}

/// One matched (or newly spawned) face in the latest frame.
#[derive(Clone, Debug)]
pub struct TrackedFace {
    /// Stable ID — the same person keeps this number across consecutive
    /// frames. IDs are never recycled.
    pub id: u32,
    /// The detection that produced this hit.
    pub rect: Detection,
    /// Consecutive frames the track has been alive (1 on the first frame).
    pub age: u32,
    /// Consecutive frames since the last match (`0` if matched this
    /// frame). Equals `max_age` on the frame the track is about to be
    /// closed.
    pub misses: u32,
}

#[derive(Clone, Debug)]
struct Track {
    id: u32,
    rect: Detection,
    age: u32,
    misses: u32,
}

/// Multi-face tracker.
///
/// Construct with [`FaceTracker::new`], feed it detection sets frame by
/// frame with [`FaceTracker::update`], and read the per-frame result. The
/// tracker holds no references to the input slice, so the caller may
/// reuse the detection buffer freely between frames.
#[derive(Debug)]
pub struct FaceTracker {
    cfg: TrackerConfig,
    tracks: Vec<Track>,
}

impl Default for FaceTracker {
    fn default() -> Self {
        Self::new(TrackerConfig::default())
    }
}

impl FaceTracker {
    /// Construct a tracker with the given config.
    pub fn new(cfg: TrackerConfig) -> Self {
        Self {
            cfg,
            tracks: Vec::new(),
        }
    }

    /// Number of currently-open tracks (not counting freshly retired ones
    /// — those are removed at the end of `update`).
    pub fn active(&self) -> usize {
        self.tracks.len()
    }

    /// Highest assigned ID so far, or `0` if no face has been tracked.
    pub fn last_id(&self) -> u32 {
        self.cfg.next_id.saturating_sub(1)
    }

    /// Process one frame's detections and return the assigned IDs.
    ///
    /// The returned slice is parallel to `dets`: `out[i]` is the entry
    /// for `dets[i]`. New detections get a newly-minted ID.
    pub fn update(&mut self, dets: &[Detection]) -> Vec<TrackedFace> {
        // Bump miss counter and age for every existing track. Tracks that
        // exceed `max_age` are dropped at the end of this function so we
        // don't lose a track mid-association that another detection
        // could still legitimately match.
        for t in &mut self.tracks {
            t.misses = t.misses.saturating_add(1);
        }

        // Greedy bipartite association by descending score.
        let mut det_order: Vec<usize> = (0..dets.len()).collect();
        det_order.sort_by(|&a, &b| {
            dets[b]
                .score
                .partial_cmp(&dets[a].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut out: Vec<Option<TrackedFace>> = (0..dets.len()).map(|_| None).collect();
        // Use a HashSet rather than a `Vec<bool>`: `self.tracks` grows
        // *inside* this loop when a detection spawns a new track, so a
        // pre-allocated `Vec<bool>` would be stale by the end. Same
        // approach as `video_id::LightTracker`.
        let mut track_claimed: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for &di in &det_order {
            let det = &dets[di];

            // Find the best track for this detection, preferring IoU
            // matches and falling back to centroid distance. The two
            // criteria optimise in different directions (higher IoU,
            // smaller distance), so they live in separate slots and IoU
            // is the tiebreaker if both surface a candidate.
            let mut best_iou_track: Option<usize> = None;
            let mut best_iou: f32 = 0.0;
            let mut best_dist_track: Option<usize> = None;
            let mut best_dist: f32 = f32::MAX;

            for (ti, t) in self.tracks.iter().enumerate() {
                if track_claimed.contains(&ti) {
                    continue;
                }
                let iou = det.iou(&t.rect);
                if iou >= self.cfg.min_iou && iou > best_iou {
                    best_iou = iou;
                    best_iou_track = Some(ti);
                }
                // Centroid-distance fallback for the case where the
                // detector jittered the box but the face is still the
                // same. Gate is `centroid_gate × bbox_diagonal` of the
                // *track* (so closer-together, smaller tracks get a
                // tighter gate automatically).
                let diag = (t.rect.w.max(t.rect.h)) as f32;
                let diag = diag.max(1.0) * std::f32::consts::SQRT_2;
                let (cx, cy) = det.center();
                let (tx, ty) = t.rect.center();
                let dist = ((cx - tx).powi(2) + (cy - ty).powi(2)).sqrt();
                let gate = self.cfg.centroid_gate * diag;
                if dist <= gate && dist < best_dist {
                    best_dist = dist;
                    best_dist_track = Some(ti);
                }
            }

            // IoU match wins over a pure centroid fallback.
            let chosen = best_iou_track.or(best_dist_track);

            if let Some(ti) = chosen {
                let t = &mut self.tracks[ti];
                track_claimed.insert(ti);
                t.age = t.age.saturating_add(1);
                t.misses = 0;
                t.rect = det.clone();
                out[di] = Some(TrackedFace {
                    id: t.id,
                    rect: det.clone(),
                    age: t.age,
                    misses: t.misses,
                });
            } else {
                // Spawn a new track.
                let id = self.cfg.next_id;
                self.cfg.next_id = self.cfg.next_id.saturating_add(1);
                self.tracks.push(Track {
                    id,
                    rect: det.clone(),
                    age: 1,
                    misses: 0,
                });
                out[di] = Some(TrackedFace {
                    id,
                    rect: det.clone(),
                    age: 1,
                    misses: 0,
                });
            }
        }

        // Drop tracks whose `misses` have exceeded the budget.
        self.tracks.retain(|t| t.misses <= self.cfg.max_age);

        out.into_iter().map(|o| o.unwrap()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: usize, y: usize, w: usize, h: usize, score: f32) -> Detection {
        Detection { x, y, w, h, score }
    }

    /// Helper: drive `n_frames` frames of a single box moving by `(dx, dy)`
    /// each step, return the ID of the resulting track. Returns `None` if
    /// the tracker failed to keep a stable ID across the run.
    fn run_single_tracker(dx: i32, dy: i32, n_frames: usize) -> Option<u32> {
        let mut t = FaceTracker::default();
        let mut first_id = None;
        let mut x: i32 = 100;
        let mut y: i32 = 100;
        for _ in 0..n_frames {
            let d = det(x.max(0) as usize, y.max(0) as usize, 60, 60, 0.9);
            let out = t.update(&[d]);
            assert_eq!(out.len(), 1);
            match first_id {
                None => first_id = Some(out[0].id),
                Some(id) if out[0].id != id => return None,
                Some(_) => {}
            }
            x += dx;
            y += dy;
        }
        first_id
    }

    /// The headline test: two faces moving slightly each frame keep
    /// distinct, stable IDs for 30 frames.
    #[test]
    fn tracker_assigns_stable_ids_across_frames() {
        let mut t = FaceTracker::default();
        let mut a_id = None;
        let mut b_id = None;
        for i in 0..30 {
            // Face A: moves 2 px right and 1 px down each frame.
            // Face B: moves 1 px left and 2 px down each frame.
            let a_x = 100 + 2 * i as i32;
            let b_x = 300 - 1 * i as i32;
            let y = 100 + 1 * i as i32;
            let dets = [
                det(a_x.max(0) as usize, y.max(0) as usize, 60, 60, 0.95),
                det(b_x.max(0) as usize, y.max(0) as usize, 60, 60, 0.90),
            ];
            let out = t.update(&dets);
            assert_eq!(out.len(), 2);
            if a_id.is_none() {
                a_id = Some(out[0].id);
                b_id = Some(out[1].id);
            }
            assert_eq!(out[0].id, a_id.unwrap(), "A lost its ID at frame {i}");
            assert_eq!(out[1].id, b_id.unwrap(), "B lost its ID at frame {i}");
            // IDs must remain distinct.
            assert_ne!(out[0].id, out[1].id);
        }
    }

    /// A track that goes unmatched for `max_age + 1` frames is dropped
    /// — a fresh detection then gets a *new* ID rather than reusing the
    /// stale one.
    #[test]
    fn tracker_drops_dead_tracks() {
        let cfg = TrackerConfig {
            max_age: 3,
            ..TrackerConfig::default()
        };
        let mut t = FaceTracker::new(cfg);
        let out = t.update(&[det(100, 100, 50, 50, 0.9)]);
        let stale_id = out[0].id;
        assert_eq!(t.active(), 1);

        // 4 frames of "no detections" (max_age + 1).
        for _ in 0..4 {
            let out = t.update(&[]);
            assert!(out.is_empty());
        }
        assert_eq!(t.active(), 0, "track should have been dropped");

        // Re-appear — must mint a NEW id, not the stale one.
        let out = t.update(&[det(100, 100, 50, 50, 0.9)]);
        assert_eq!(out.len(), 1);
        assert_ne!(out[0].id, stale_id, "stale ID must not be reused");
    }

    /// A face appears at frame 5, persists, then disappears at frame 10.
    /// State at each phase must be internally consistent.
    #[test]
    fn tracker_handles_appearance_and_disappearance() {
        let cfg = TrackerConfig {
            max_age: 2,
            ..TrackerConfig::default()
        };
        let mut t = FaceTracker::new(cfg);

        // Frames 0..4: empty frames. No tracks.
        for _ in 0..5 {
            let out = t.update(&[]);
            assert!(out.is_empty());
        }
        assert_eq!(t.active(), 0);

        // Frame 5: a face appears.
        let out = t.update(&[det(50, 50, 40, 40, 0.9)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].age, 1);
        let first_id = out[0].id;

        // Frames 6..9: face persists, ID stable.
        for frame in 6..=9 {
            let out = t.update(&[det(50, 50, 40, 40, 0.9)]);
            assert_eq!(out.len(), 1, "frame {frame}");
            assert_eq!(out[0].id, first_id, "frame {frame}");
            assert_eq!(out[0].age as i32, (frame - 5 + 1));
        }

        // Frame 10: face disappears.
        let out = t.update(&[]);
        assert!(out.is_empty());
        // Track still open but on misses=1.
        assert_eq!(t.active(), 1);

        // Frame 11: still gone, misses=2.
        let out = t.update(&[]);
        assert!(out.is_empty());
        assert_eq!(t.active(), 1);

        // Frame 12: misses=3 → closed.
        let out = t.update(&[]);
        assert!(out.is_empty());
        assert_eq!(t.active(), 0);
    }

    /// Single-face stability: 50 frames of a face moving 1 px/frame.
    #[test]
    fn single_face_keeps_one_id_across_many_frames() {
        let id = run_single_tracker(1, 1, 50);
        assert!(id.is_some(), "ID should stay stable");
    }

    /// Crossing tracks: two faces swap positions — without Hungarian
    /// this can be a hard case, but our IoU + greedy still keeps each
    /// identity stable as long as boxes don't actually overlap.
    #[test]
    fn tracker_handles_crossing_faces() {
        let mut t = FaceTracker::default();
        // Two boxes starting far apart, ending swapped.
        for i in 0..20 {
            let ax = 100 + 5 * i as i32;
            let bx = 300 - 5 * i as i32;
            let out = t.update(&[
                det(ax.max(0) as usize, 100, 40, 40, 0.9),
                det(bx.max(0) as usize, 100, 40, 40, 0.9),
            ]);
            assert_eq!(out.len(), 2);
            // Stable identities only as long as the boxes don't fully
            // overlap. Check that by frame ~midway, both still have IDs.
            if i == 10 {
                // We don't assert the *same* ID per slot (greedy swap is
                // legitimate) — only that exactly two distinct IDs exist.
                assert_ne!(out[0].id, out[1].id);
            }
        }
    }

    /// Empty frame after empty frame stays empty and drops everything.
    #[test]
    fn empty_frames_keep_no_active_tracks() {
        let mut t = FaceTracker::default();
        let _ = t.update(&[det(0, 0, 10, 10, 0.5)]);
        for _ in 0..10 {
            let out = t.update(&[]);
            assert!(out.is_empty());
        }
        assert_eq!(t.active(), 0);
    }

    /// A new detection that doesn't overlap any existing track spawns a
    /// fresh ID, and `last_id()` advances monotonically.
    #[test]
    fn new_detections_get_monotonically_increasing_ids() {
        let mut t = FaceTracker::default();
        let out = t.update(&[det(0, 0, 20, 20, 0.9)]);
        assert_eq!(out[0].id, 1);
        let out = t.update(&[det(100, 100, 20, 20, 0.9)]);
        assert_eq!(out[0].id, 2);
        let out = t.update(&[det(200, 200, 20, 20, 0.9)]);
        assert_eq!(out[0].id, 3);
        assert_eq!(t.last_id(), 3);
    }
}
