//! Run the zero-dep 5-point landmark estimator on a synthetic frontal
//! face and print the result.
//!
//! Demonstrates:
//! - building a synthetic image with five known landmark positions,
//! - calling [`rsface::landmarks::estimate_five_point`] with a bbox,
//! - packing the result into the canonical
//!   [`rsface::face::Landmarks`] used by ArcFace alignment.
//!
//! Run with:
//!   cargo run --release --example landmarks_demo

use rsface::face::Landmarks;
use rsface::image::GrayImage;
use rsface::landmarks::{estimate_five_point, Rect};

fn synthetic_face() -> (GrayImage, [(f32, f32); 5]) {
    let mut img = GrayImage::new(64, 64);
    // Skin tone background.
    for y in 0..64 {
        for x in 0..64 {
            img[(x, y)] = 180;
        }
    }
    // Left eye — 5x5 dark patch centred on (20, 25).
    for dy in 0..5 {
        for dx in 0..5 {
            img[(18 + dx, 23 + dy)] = 50;
        }
    }
    // Right eye.
    for dy in 0..5 {
        for dx in 0..5 {
            img[(42 + dx, 23 + dy)] = 50;
        }
    }
    // Nose — 3-wide, 5-tall dark vertical patch centred on (32, 35).
    for dy in 0..5 {
        for dx in 0..3 {
            img[(31 + dx, 33 + dy)] = 80;
        }
    }
    // Mouth — horizontal dark line at y=46.
    for x in 22..=42 {
        img[(x, 46)] = 60;
    }

    let gt = [
        (20.0, 25.0), // left eye
        (44.0, 25.0), // right eye
        (32.0, 35.0), // nose
        (22.0, 46.0), // left mouth
        (42.0, 46.0), // right mouth
    ];
    (img, gt)
}

fn main() {
    let (img, ground_truth) = synthetic_face();
    let bbox = Rect {
        x: 0,
        y: 0,
        w: 64,
        h: 64,
    };

    let lms = estimate_five_point(&img, bbox);

    println!("=== 5-point landmark estimator demo ===");
    println!(
        "estimated inter-ocular distance: {:.2} px",
        lms.eye_distance()
    );
    println!();
    println!(
        "{:<10} {:>14} {:>14} {:>8} {:>8}",
        "feature", "estimated (x,y)", "ground truth", "dx", "dy"
    );
    println!("{}", "-".repeat(60));

    let names = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"];
    let est: [(f32, f32); 5] = [
        lms.left_eye,
        lms.right_eye,
        lms.nose,
        lms.left_mouth,
        lms.right_mouth,
    ];
    for i in 0..5 {
        let (ex, ey) = est[i];
        let (gx, gy) = ground_truth[i];
        println!(
            "{:<10} ({:>5.1}, {:>5.1}) ({:>5.1}, {:>5.1}) {:>8.2} {:>8.2}",
            names[i],
            ex,
            ey,
            gx,
            gy,
            (ex - gx).abs(),
            (ey - gy).abs(),
        );
    }

    // Pack into the canonical Landmarks type used by alignment.
    let _: Landmarks = lms.to_face_landmarks();
    println!();
    println!("packed into `face::Landmarks` ready for `align::norm_crop`.");
}
