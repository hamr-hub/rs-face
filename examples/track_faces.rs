//! Multi-face tracker demo.
//!
//! Drives [`rsface::tracker::FaceTracker`] with five synthetic frames
//! containing two moving boxes, prints the assigned IDs each frame, and
//! shows the active-track count after each update.
//!
//! Run with:
//!   cargo run --release --example track_faces

use rsface::face::Detection;
use rsface::tracker::{FaceTracker, TrackerConfig};

fn det(x: usize, y: usize, w: usize, h: usize, score: f32) -> Detection {
    Detection { x, y, w, h, score }
}

fn main() {
    let cfg = TrackerConfig::default();
    let mut tracker = FaceTracker::new(cfg);

    println!("=== Multi-face tracker demo (5 frames) ===");
    println!();
    println!(
        "{:<6} {:<22} {:<10} {:<8}",
        "frame", "boxes (x,y,w,h) × score", "ids", "active"
    );
    println!("{}", "-".repeat(60));

    // Frame 0: two faces, far apart.
    let frame0 = [det(80, 60, 60, 60, 0.95), det(280, 120, 60, 60, 0.90)];
    let out = tracker.update(&frame0);
    print_frame(0, &frame0, &out, tracker.active());

    // Frame 1: each face moves slightly.
    let frame1 = [det(84, 62, 60, 60, 0.93), det(274, 124, 60, 60, 0.91)];
    let out = tracker.update(&frame1);
    print_frame(1, &frame1, &out, tracker.active());

    // Frame 2: face A accelerates, face B stays put.
    let frame2 = [det(96, 70, 60, 60, 0.96), det(274, 124, 60, 60, 0.92)];
    let out = tracker.update(&frame2);
    print_frame(2, &frame2, &out, tracker.active());

    // Frame 3: a NEW face appears on the right.
    let frame3 = [
        det(100, 72, 60, 60, 0.94),
        det(274, 124, 60, 60, 0.93),
        det(420, 200, 60, 60, 0.80),
    ];
    let out = tracker.update(&frame3);
    print_frame(3, &frame3, &out, tracker.active());

    // Frame 4: middle face disappears, the other two continue.
    let frame4 = [det(104, 74, 60, 60, 0.95), det(424, 202, 60, 60, 0.85)];
    let out = tracker.update(&frame4);
    print_frame(4, &frame4, &out, tracker.active());

    println!();
    println!("last_id assigned: {}", tracker.last_id());
}

fn print_frame(
    frame_idx: usize,
    dets: &[Detection],
    out: &[rsface::tracker::TrackedFace],
    active: usize,
) {
    let mut summary = String::new();
    for (i, d) in dets.iter().enumerate() {
        if i > 0 {
            summary.push_str("  ");
        }
        summary.push_str(&format!("({},{},{}x{}) {:.2}", d.x, d.y, d.w, d.h, d.score));
    }
    let ids: Vec<String> = out.iter().map(|t| t.id.to_string()).collect();
    println!(
        "{:<6} {:<22} {:<10} {:<8}",
        frame_idx,
        summary,
        ids.join(","),
        active,
    );
}
