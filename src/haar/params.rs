//! Built-in demo cascade + cascade construction helpers.
//!
//! The full Viola-Jones face detector uses ~6 000 features across ~25 stages,
//! trained with AdaBoost on thousands of labeled face / non-face samples.
//! Replicating that from scratch here is out of scope; what we *do* provide is:
//!
//! 1. [`demo_face_cascade()`] — a small hand-crafted 4-stage cascade that
//!    detects a "bright center, darker border" elliptical pattern. It is
//!    calibrated to match the synthetic test source [`crate::source::synthetic`]
//!    and is useful for verifying the whole pipeline end-to-end.
//!
//! 2. A documented binary cascade file format ([`Cascade::save`]) so users can
//!    train their own cascades (e.g., convert OpenCV's XML) and load them via
//!    [`Cascade::load`].
//!
//! To convert an OpenCV `haarcascade_frontalface_default.xml` to our format,
//! see the `tools/convert_opencv.py` script in the project.

use super::cascade::{Cascade, Stage, WeakFeature};
use super::feature::{FeatureKind, HaarFeature, Rect};

/// Construct a more realistic demo cascade over a 24 × 24 window.
///
/// This is still a hand-tuned cascade (we do not have OpenCV's trained XML
/// in-tree to keep zero deps), but it covers the geometric and photometric
/// cues that a real face cascade typically uses:
///
/// 1. Bright centre band (cheek/nose) versus darker forehead + chin.
/// 2. Bright centre band versus darker side regions (left/right hair).
/// 3. Eye band darker than forehead and cheek regions.
/// 4. Nose bridge brighter than left/right cheeks.
/// 5. Strong vertical edge between forehead and mid-face (hairline).
/// 6. Strong vertical edge between mid-face and chin (jaw shadow).
/// 7. Mid-line symmetry: left and right halves similarly bright.
/// 8. Top-half brightness similar to bottom-half (face, not edge of body).
/// 9. Vertical symmetry around the centre column.
/// 10. Strong horizontal gradient at cheek height (face vs background).
///
/// The thresholds are tuned for the synthetic face-like pattern used in our
/// tests; for real photographs load a trained `.rfcf` cascade built from
/// OpenCV's `haarcascade_frontalface_default.xml` via
/// `tools/convert_opencv_xml.py`.
pub fn demo_face_cascade() -> Cascade {
    let mut c = Cascade::new(24, 24);

    // Build a small bank of useful features.
    let f_vert_center_full = HaarFeature {
        kind: FeatureKind::VerticalCenter,
        width: 1,
        height: 3,
        rects: vec![
            Rect::new(0, 0, 1, 1, 1.0),
            Rect::new(0, 1, 1, 1, -2.0),
            Rect::new(0, 2, 1, 1, 1.0),
        ],
    };
    let f_horiz_center_full = HaarFeature {
        kind: FeatureKind::HorizontalCenter,
        width: 3,
        height: 1,
        rects: vec![
            Rect::new(0, 0, 1, 1, 1.0),
            Rect::new(1, 0, 1, 1, -2.0),
            Rect::new(2, 0, 1, 1, 1.0),
        ],
    };
    let f_vert_edge_full = HaarFeature::vertical_edge(1, 2);
    let f_horiz_edge_full = HaarFeature::horizontal_edge(2, 1);

    // Eye-band feature: top 30%, mid 40%, bottom 30% vertical layout. Eye band
    // is typically slightly darker than forehead and cheeks.
    let f_eye_band = HaarFeature {
        kind: FeatureKind::VerticalCenter,
        width: 1,
        height: 5,
        rects: vec![
            Rect::new(0, 0, 1, 1, 1.0),
            Rect::new(0, 1, 1, 1, -1.0),
            Rect::new(0, 2, 1, 1, -1.0),
            Rect::new(0, 3, 1, 1, -1.0),
            Rect::new(0, 4, 1, 1, 1.0),
        ],
    };

    // Nose-bridge feature: 3-column horizontal center-surround, but with
    // weighted mid column to detect a vertical bright streak.
    let f_nose_bridge = HaarFeature {
        kind: FeatureKind::HorizontalCenter,
        width: 3,
        height: 1,
        rects: vec![
            Rect::new(0, 0, 1, 1, 1.0),
            Rect::new(1, 0, 1, 1, -3.0),
            Rect::new(2, 0, 1, 1, 1.0),
        ],
    };

    // Top-half vs bottom-half: 1x2 vertical edge covering the upper region.
    let f_top_bottom = HaarFeature {
        kind: FeatureKind::VerticalEdge,
        width: 1,
        height: 2,
        rects: vec![Rect::new(0, 0, 1, 1, 1.0), Rect::new(0, 1, 1, 1, -1.0)],
    };

    // Left-half vs right-half: horizontal edge for vertical symmetry.
    let f_left_right = HaarFeature {
        kind: FeatureKind::HorizontalEdge,
        width: 2,
        height: 1,
        rects: vec![Rect::new(0, 0, 1, 1, 1.0), Rect::new(1, 0, 1, 1, -1.0)],
    };

    let all = vec![
        f_vert_center_full,
        f_horiz_center_full,
        f_vert_edge_full,
        f_horiz_edge_full,
        f_eye_band,
        f_nose_bridge,
        f_top_bottom,
        f_left_right,
    ];
    let idx_of = |f: &HaarFeature| -> usize {
        all.iter()
            .position(|x| std::ptr::eq(x, f))
            .expect("feature not in bank")
    };
    let i_vc = idx_of(&all[0]);
    let i_hc = idx_of(&all[1]);
    let i_ve = idx_of(&all[2]);
    let i_he = idx_of(&all[3]);
    let i_eye = idx_of(&all[4]);
    let i_nose = idx_of(&all[5]);
    let i_tb = idx_of(&all[6]);
    let i_lr = idx_of(&all[7]);

    c.features.extend(all);

    // Stage 1 — strong "bright center" cue. The vertical-center-surround
    // response is *negative* when the centre is brighter than top/bottom.
    // Eval convention: value < threshold → left_val (face), else → right_val.
    c.stages.push(Stage {
        stage_threshold: 0.5,
        weak_features: vec![WeakFeature {
            feature_index: i_vc as u32,
            threshold: 0.0,
            sign: 1,
            left_val: 1.5,
            right_val: -1.0,
        }],
    });

    // Stage 2 — bright center vs dark sides.
    c.stages.push(Stage {
        stage_threshold: 0.5,
        weak_features: vec![WeakFeature {
            feature_index: i_hc as u32,
            threshold: 0.0,
            sign: 1,
            left_val: 1.0,
            right_val: -1.0,
        }],
    });

    // Stage 3 — eye band darker than forehead + chin.
    c.stages.push(Stage {
        stage_threshold: -1.0,
        weak_features: vec![WeakFeature {
            feature_index: i_eye as u32,
            threshold: 0.0,
            sign: 1,
            left_val: 0.8,
            right_val: -0.6,
        }],
    });

    // Stage 4 — vertical edge between top and bottom halves.
    c.stages.push(Stage {
        stage_threshold: -1.0,
        weak_features: vec![WeakFeature {
            feature_index: i_ve as u32,
            threshold: 0.0,
            sign: 1,
            left_val: 0.6,
            right_val: -0.5,
        }],
    });

    // Stage 5 — nose bridge brightness.
    c.stages.push(Stage {
        stage_threshold: -1.0,
        weak_features: vec![WeakFeature {
            feature_index: i_nose as u32,
            threshold: 0.0,
            sign: 1,
            left_val: 0.5,
            right_val: -0.4,
        }],
    });

    // Stage 6 — top vs bottom.
    c.stages.push(Stage {
        stage_threshold: -1.0,
        weak_features: vec![WeakFeature {
            feature_index: i_tb as u32,
            threshold: 0.0,
            sign: 1,
            left_val: 0.4,
            right_val: -0.3,
        }],
    });

    // Stage 7 — vertical symmetry. Threshold slightly positive so a
    // perfectly symmetric window (raw response == 0 → value ≈ 0 after
    // variance normalisation) hits the face branch.
    c.stages.push(Stage {
        stage_threshold: -1.0,
        weak_features: vec![WeakFeature {
            feature_index: i_lr as u32,
            threshold: 0.01,
            sign: 1,
            left_val: 0.3,
            right_val: -0.3,
        }],
    });

    // Stage 8 — combine the strongest cues for a final pass.
    c.stages.push(Stage {
        stage_threshold: -0.5,
        weak_features: vec![
            WeakFeature {
                feature_index: i_ve as u32,
                threshold: 0.0,
                sign: 1,
                left_val: 0.3,
                right_val: -0.2,
            },
            WeakFeature {
                feature_index: i_hc as u32,
                threshold: 0.0,
                sign: 1,
                left_val: 0.2,
                right_val: -0.2,
            },
        ],
    });

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::GrayImage;
    use crate::integral::{IntegralImage, RotatedIntegralImage};

    #[test]
    fn demo_cascade_loads() {
        let c = demo_face_cascade();
        assert!(c.num_stages() >= 3);
        assert!(c.num_features() >= 3);
    }

    #[test]
    fn demo_cascade_save_load_roundtrip() {
        let c = demo_face_cascade();
        let dir = std::env::temp_dir().join("rsface_cascade_test");
        let path = dir.join("demo.rfcf");
        std::fs::create_dir_all(&dir).unwrap();
        c.save(&path).unwrap();
        let back = Cascade::load(&path).unwrap();
        assert_eq!(back.num_stages(), c.num_stages());
        assert_eq!(back.num_features(), c.num_features());
        assert_eq!(back.window_w, c.window_w);
        assert_eq!(back.window_h, c.window_h);
    }

    #[test]
    fn demo_classifies_bright_center_pattern() {
        // Construct a 24x24 image simulating a face: bright forehead band at top,
        // dark "hair" upper region, bright center "face", dark lower region.
        let mut img = GrayImage::new(24, 24);
        for y in 0..24 {
            for x in 0..24 {
                let v = if y < 4 {
                    20
                }
                // top "sky" / dark
                else if y < 8 && (4..20).contains(&x) {
                    200
                }
                // bright forehead band
                else if y < 18 && (4..20).contains(&x) {
                    220
                }
                // bright face oval
                else {
                    20
                }; // dark border
                img[(x, y)] = v;
            }
        }
        let ii = IntegralImage::from_gray(&img);
        let ri = RotatedIntegralImage::from_gray(&img);
        let c = demo_face_cascade();
        let mut cache = crate::haar::EvalCache::new(c.features.len());
        let res = c.classify(&ii, &ri, 0, 0, &mut cache);
        assert!(
            res.is_some(),
            "demo cascade should accept bright-center pattern"
        );
    }
}
