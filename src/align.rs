//! Face alignment: 5-point similarity transform onto the ArcFace canonical layout.
//!
//! Every modern face-recognition backbone (ArcFace, AdaFace, CosFace, …) is trained on
//! crops that have been *warped* so the five landmarks land on fixed canonical
//! positions. Embeddings are only comparable between two faces if both went through
//! the same warp. Skipping alignment — or cropping the raw detector box and resizing —
//! is the single most common cause of a recognition pipeline that "runs fine" while
//! producing cosine similarities barely above chance.
//!
//! This module implements the closed-form 2D similarity (Procrustes / Umeyama) fit that
//! InsightFace's `norm_crop` performs, and a bilinear inverse warp to apply it.
//!
//! Note the transform is a *similarity*, not a full affine: it has 4 degrees of freedom
//! (rotation, isotropic scale, translation) and deliberately cannot shear or stretch
//! anisotropically. A full 6-DoF affine fit through 5 noisy landmarks would happily
//! distort face geometry to reduce landmark residual, changing the very proportions the
//! embedding encodes.

use crate::face::{Landmarks, NUM_LANDMARKS};
use crate::image::RgbImage;

/// Canonical output size of an ArcFace-family crop, in pixels (square).
pub const ARCFACE_CROP_SIZE: usize = 112;

/// The canonical 5-point landmark layout ArcFace was trained against, for a
/// 112x112 crop, in `(x, y)` pixel coordinates and [`crate::face::landmark`] order.
///
/// These exact constants come from InsightFace's `arcface_dst` reference
/// (`insightface/utils/face_align.py`) and must not be "tidied" — they are asymmetric by
/// design (the eyes sit at y=51.70 and y=51.50, not a shared y), because they are the
/// empirical mean landmark configuration of the training set rather than a hand-drawn
/// ideal.
///
/// # The 96-vs-112 trap
///
/// Older InsightFace code and much third-party documentation quote a *different* array
/// whose x values are exactly 8.0 smaller: `[30.2946, 65.5318, 48.0252, 33.5493,
/// 62.7299]`. That is the legacy **96x112** template. `face_align.py` historically
/// applied `src[:, 0] += 8.0` when `mode == 'arcface'` to recentre it for a 112-wide
/// crop, and modern versions bake the result in as `arcface_dst` — the values below.
///
/// Using the 96-wide numbers with a 112x112 output shifts every aligned face 8 px left.
/// Nothing errors: crops still look like faces and embeddings still normalise, but every
/// one is computed off-centre relative to the backbone's training distribution, quietly
/// costing accuracy. The sanity check is that mean x must equal `112/2`, which
/// `reference_landmarks_are_centred_for_112` asserts.
pub const ARCFACE_REFERENCE_LANDMARKS: [(f32, f32); NUM_LANDMARKS] = [
    (38.2946, 51.6963), // left eye
    (73.5318, 51.5014), // right eye
    (56.0252, 71.7366), // nose
    (41.5493, 92.3655), // left mouth corner
    (70.7299, 92.2041), // right mouth corner
];

/// A 2D similarity transform stored as the 2x3 matrix
/// `[[a, -b, tx], [b, a, ty]]`, mapping source pixels to destination pixels.
///
/// Only two rotation/scale parameters are stored (`a`, `b`) rather than four, which
/// makes it structurally impossible to represent a shear — the type enforces the
/// 4-DoF constraint that the algorithm requires.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimilarityTransform {
    pub a: f32,
    pub b: f32,
    pub tx: f32,
    pub ty: f32,
}

impl SimilarityTransform {
    /// The identity transform.
    pub const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        tx: 0.0,
        ty: 0.0,
    };

    /// Uniform scale factor of this transform.
    #[inline]
    pub fn scale(&self) -> f32 {
        (self.a * self.a + self.b * self.b).sqrt()
    }

    /// Rotation angle in radians.
    #[inline]
    pub fn rotation(&self) -> f32 {
        self.b.atan2(self.a)
    }

    /// Apply the transform to a point.
    #[inline]
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.a * x - self.b * y + self.tx,
            self.b * x + self.a * y + self.ty,
        )
    }

    /// Invert the transform.
    ///
    /// Returns `None` when the transform is degenerate (zero scale), which happens if
    /// the detector emitted five coincident landmarks. Returning `None` rather than
    /// dividing by ~0 keeps a NaN-poisoned crop out of the embedding model.
    pub fn inverse(&self) -> Option<Self> {
        let det = self.a * self.a + self.b * self.b;
        if det < 1e-12 {
            return None;
        }
        let ia = self.a / det;
        let ib = -self.b / det;
        // Inverse translation is -R^-1 * t.
        Some(Self {
            a: ia,
            b: ib,
            tx: -(ia * self.tx - ib * self.ty),
            ty: -(ib * self.tx + ia * self.ty),
        })
    }
}

/// Estimate the least-squares 2D similarity transform mapping `src` onto `dst`.
///
/// This is the closed-form Umeyama solution specialised to 2D: with the point sets
/// centred, the optimal rotation and scale fall out of two dot-product accumulators, so
/// there is no iteration and no SVD required. Being closed-form matters here because
/// this runs once per detected face per frame.
///
/// Returns `None` if `src` is degenerate (all points coincident).
pub fn estimate_similarity(
    src: &[(f32, f32); NUM_LANDMARKS],
    dst: &[(f32, f32); NUM_LANDMARKS],
) -> Option<SimilarityTransform> {
    let n = NUM_LANDMARKS as f32;

    let (mut sx, mut sy, mut dx, mut dy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..NUM_LANDMARKS {
        sx += src[i].0;
        sy += src[i].1;
        dx += dst[i].0;
        dy += dst[i].1;
    }
    let (sx, sy, dx, dy) = (sx / n, sy / n, dx / n, dy / n);

    // Accumulate centred cross-terms.
    //   num_a  = sum <q_i, p_i>      -> s*cos(theta) numerator
    //   num_b  = sum cross(p_i, q_i) -> s*sin(theta) numerator
    //   var_src = sum |p_i|^2
    let mut num_a = 0.0f32;
    let mut num_b = 0.0f32;
    let mut var_src = 0.0f32;
    for i in 0..NUM_LANDMARKS {
        let (px, py) = (src[i].0 - sx, src[i].1 - sy);
        let (qx, qy) = (dst[i].0 - dx, dst[i].1 - dy);
        num_a += qx * px + qy * py;
        num_b += qy * px - qx * py;
        var_src += px * px + py * py;
    }

    if var_src < 1e-12 {
        return None;
    }

    let a = num_a / var_src;
    let b = num_b / var_src;

    Some(SimilarityTransform {
        a,
        b,
        tx: dx - (a * sx - b * sy),
        ty: dy - (b * sx + a * sy),
    })
}

/// Estimate the transform taking a detected face's landmarks to the ArcFace canonical
/// 112x112 layout.
pub fn estimate_arcface_transform(lms: &Landmarks) -> Option<SimilarityTransform> {
    estimate_similarity(&lms.points, &ARCFACE_REFERENCE_LANDMARKS)
}

/// How [`warp_similarity_fill`] colours destination pixels whose inverse-mapped
/// sample lies outside the source image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BorderMode {
    /// Fill with a constant colour — the `borderValue=0.0` behaviour of the
    /// OpenCV `warpAffine` call in InsightFace's `norm_crop`. ArcFace-family
    /// backbones were trained on black-bordered crops, so this is the default
    /// and required for embedding parity.
    Constant([u8; 3]),
    /// Replicate the nearest edge pixel (`BORDER_REPLICATE`). Useful for
    /// visualisation and for detectors fed back on their own crops, but NOT
    /// what the ArcFace training distribution looks like.
    Replicate,
}

/// Bilinearly warp `src` through `t` into a `out_w` x `out_h` RGB image,
/// filling out-of-bounds samples per `border` (see [`BorderMode`]).
///
/// Implemented as an *inverse* warp: we iterate destination pixels and pull from the
/// source. A forward warp would scatter, leaving unwritten holes wherever the transform
/// magnifies — a classic source of speckled crops that quietly degrade embeddings.
///
/// Coordinate convention matches OpenCV `warpAffine` exactly: destination pixel
/// `(ox, oy)` (top-left origin, the same system detector landmarks are reported in)
/// samples the source at `t⁻¹(ox, oy)` — there is deliberately **no** +0.5 pixel-centre
/// shift. Adding one would translate every aligned crop by half a pixel relative to the
/// canonical landmark layout the backbone was trained against.
pub fn warp_similarity_fill(
    src: &RgbImage,
    t: &SimilarityTransform,
    out_w: usize,
    out_h: usize,
    border: BorderMode,
) -> Option<RgbImage> {
    let inv = t.inverse()?;
    let (sw, sh) = (src.width(), src.height());
    if sw == 0 || sh == 0 {
        return None;
    }

    let mut out = RgbImage::new(out_w, out_h);
    let max_x = (sw - 1) as f32;
    let max_y = (sh - 1) as f32;
    let src_data = src.as_slice();

    // One bilinear corner: the source pixel value, or per BorderMode a fill
    // when the integer corner falls outside the image.
    let sample = |x: isize, y: isize, ch: usize| -> f32 {
        if x >= 0 && y >= 0 && (x as usize) < sw && (y as usize) < sh {
            src_data[(y as usize * sw + x as usize) * 3 + ch] as f32
        } else {
            match border {
                BorderMode::Constant(col) => col[ch] as f32,
                // Clamp the corner to the nearest in-range pixel.
                BorderMode::Replicate => {
                    let cx = x.clamp(0, max_x as isize) as usize;
                    let cy = y.clamp(0, max_y as isize) as usize;
                    src_data[(cy * sw + cx) * 3 + ch] as f32
                }
            }
        }
    };

    for oy in 0..out_h {
        for ox in 0..out_w {
            let (fx, fy) = inv.apply(ox as f32, oy as f32);
            // Floor indices as signed so corners just outside the frame map to
            // border pixels instead of wrapping via a usize cast.
            let x0 = fx.floor() as isize;
            let y0 = fy.floor() as isize;
            let wx = fx - x0 as f32;
            let wy = fy - y0 as f32;
            let o = (oy * out_w + ox) * 3;

            for c in 0..3 {
                let v00 = sample(x0, y0, c);
                let v10 = sample(x0 + 1, y0, c);
                let v01 = sample(x0, y0 + 1, c);
                let v11 = sample(x0 + 1, y0 + 1, c);
                let top = v00 * (1.0 - wx) + v10 * wx;
                let bot = v01 * (1.0 - wx) + v11 * wx;
                // +0.5 to round rather than truncate; truncation biases every crop dark.
                out.as_mut_slice()[o + c] = (top * (1.0 - wy) + bot * wy + 0.5) as u8;
            }
        }
    }
    Some(out)
}

/// Bilinearly warp `src` through `t` with InsightFace/OpenCV parity: black
/// (`[0,0,0]`) constant border, no half-pixel shift. This is the warp ArcFace
/// crops must use; use [`warp_similarity_fill`] with [`BorderMode::Replicate`]
/// only for visualisation.
pub fn warp_similarity(
    src: &RgbImage,
    t: &SimilarityTransform,
    out_w: usize,
    out_h: usize,
) -> Option<RgbImage> {
    warp_similarity_fill(src, t, out_w, out_h, BorderMode::Constant([0, 0, 0]))
}

/// Full `norm_crop`: align a detected face to the canonical ArcFace 112x112 crop.
///
/// This is the function the recognition pipeline should call. Returns `None` when the
/// landmarks are degenerate, so the caller can drop the face instead of embedding noise.
pub fn norm_crop(src: &RgbImage, lms: &Landmarks) -> Option<RgbImage> {
    let t = estimate_arcface_transform(lms)?;
    warp_similarity(src, &t, ARCFACE_CROP_SIZE, ARCFACE_CROP_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REF: [(f32, f32); NUM_LANDMARKS] = ARCFACE_REFERENCE_LANDMARKS;

    fn lms(points: [(f32, f32); NUM_LANDMARKS]) -> Landmarks {
        Landmarks { points }
    }

    /// Max residual between the transform applied to `src` and the expected `dst`.
    fn residual(t: &SimilarityTransform, src: &[(f32, f32); 5], dst: &[(f32, f32); 5]) -> f32 {
        let mut worst = 0.0f32;
        for i in 0..NUM_LANDMARKS {
            let (x, y) = t.apply(src[i].0, src[i].1);
            worst = worst.max(((x - dst[i].0).powi(2) + (y - dst[i].1).powi(2)).sqrt());
        }
        worst
    }

    /// Guards the 96-vs-112 template trap documented on
    /// [`ARCFACE_REFERENCE_LANDMARKS`]. The canonical 112x112 template is horizontally
    /// centred, so mean x must be 56.0. The legacy 96x112 array (every x smaller by 8.0)
    /// would centre at 48.0 and silently misalign every crop by 8 px without erroring.
    #[test]
    fn reference_landmarks_are_centred_for_112() {
        let mean_x: f32 = REF.iter().map(|p| p.0).sum::<f32>() / NUM_LANDMARKS as f32;
        assert!(
            (mean_x - ARCFACE_CROP_SIZE as f32 / 2.0).abs() < 0.05,
            "reference template mean x = {mean_x}, expected 56.0 for a 112-wide crop. \
             If this is ~48.0 the legacy 96x112 array was pasted in without the +8.0 \
             arcface recentring."
        );

        // The legacy array must NOT be what we shipped.
        const LEGACY_96: [f32; NUM_LANDMARKS] = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299];
        for (i, legacy_x) in LEGACY_96.iter().enumerate() {
            assert!(
                (REF[i].0 - legacy_x).abs() > 7.0,
                "landmark {i} matches the legacy 96x112 template"
            );
        }
    }

    /// All five points must lie inside the crop; a template point outside [0,112] would
    /// mean landmarks are being warped off-canvas.
    #[test]
    fn reference_landmarks_lie_inside_the_crop() {
        let size = ARCFACE_CROP_SIZE as f32;
        for (i, (x, y)) in REF.iter().enumerate() {
            assert!(*x > 0.0 && *x < size, "landmark {i} x={x} outside crop");
            assert!(*y > 0.0 && *y < size, "landmark {i} y={y} outside crop");
        }
    }

    /// Basic anatomy sanity: eyes above nose above mouth, right eye right of left eye.
    /// Catches a transposed or permuted template, which would otherwise produce
    /// confidently-wrong alignments.
    #[test]
    fn reference_landmarks_are_anatomically_ordered() {
        use crate::face::landmark::*;
        assert!(
            REF[LEFT_EYE].0 < REF[RIGHT_EYE].0,
            "right eye must be right of left"
        );
        assert!(REF[LEFT_EYE].1 < REF[NOSE].1, "eyes must be above nose");
        assert!(REF[NOSE].1 < REF[LEFT_MOUTH].1, "nose must be above mouth");
        assert!(
            REF[LEFT_MOUTH].0 < REF[RIGHT_MOUTH].0,
            "right mouth corner must be right of left"
        );
    }

    #[test]
    fn reference_landmarks_fit_themselves_as_identity() {
        let t = estimate_similarity(&REF, &REF).unwrap();
        assert!((t.scale() - 1.0).abs() < 1e-4, "scale={}", t.scale());
        assert!(t.rotation().abs() < 1e-4);
        assert!(residual(&t, &REF, &REF) < 1e-3);
    }

    #[test]
    fn pure_scale_is_recovered_exactly() {
        // Landmarks twice as large => transform must scale by 0.5 to normalise.
        let mut big = REF;
        for p in big.iter_mut() {
            p.0 *= 2.0;
            p.1 *= 2.0;
        }
        let t = estimate_similarity(&big, &REF).unwrap();
        assert!((t.scale() - 0.5).abs() < 1e-4, "scale={}", t.scale());
        assert!(residual(&t, &big, &REF) < 1e-3);
    }

    #[test]
    fn pure_translation_is_recovered_exactly() {
        let mut moved = REF;
        for p in moved.iter_mut() {
            p.0 += 37.0;
            p.1 -= 11.0;
        }
        let t = estimate_similarity(&moved, &REF).unwrap();
        assert!((t.scale() - 1.0).abs() < 1e-4);
        assert!(residual(&t, &moved, &REF) < 1e-3);
    }

    #[test]
    fn pure_rotation_is_recovered_exactly() {
        // Rotate the reference by +30 degrees about the origin, then fit back.
        let theta = std::f32::consts::FRAC_PI_6;
        let (c, s) = (theta.cos(), theta.sin());
        let mut rot = REF;
        for p in rot.iter_mut() {
            let (x, y) = (p.0, p.1);
            *p = (c * x - s * y, s * x + c * y);
        }
        let t = estimate_similarity(&rot, &REF).unwrap();
        assert!((t.scale() - 1.0).abs() < 1e-4);
        // Fitting back must undo the rotation.
        assert!((t.rotation() + theta).abs() < 1e-4, "rot={}", t.rotation());
        assert!(residual(&t, &rot, &REF) < 1e-3);
    }

    #[test]
    fn similarity_fit_cannot_shear() {
        // Anisotropically stretch x only. A 6-DoF affine could fit this exactly;
        // a similarity must NOT, proving we kept the 4-DoF constraint.
        let mut stretched = REF;
        for p in stretched.iter_mut() {
            p.0 *= 2.0; // x only
        }
        let t = estimate_similarity(&stretched, &REF).unwrap();
        assert!(
            residual(&t, &stretched, &REF) > 1.0,
            "a similarity transform must leave residual on a sheared input"
        );
    }

    #[test]
    fn degenerate_coincident_landmarks_return_none() {
        let same = [(50.0f32, 50.0f32); NUM_LANDMARKS];
        assert!(estimate_similarity(&same, &REF).is_none());
    }

    #[test]
    fn inverse_roundtrips_points() {
        let t = SimilarityTransform {
            a: 1.7,
            b: -0.4,
            tx: 12.0,
            ty: -5.0,
        };
        let inv = t.inverse().unwrap();
        let (x, y) = t.apply(11.0, 23.0);
        let (bx, by) = inv.apply(x, y);
        assert!((bx - 11.0).abs() < 1e-3 && (by - 23.0).abs() < 1e-3);
    }

    #[test]
    fn zero_scale_transform_has_no_inverse() {
        let t = SimilarityTransform {
            a: 0.0,
            b: 0.0,
            tx: 5.0,
            ty: 5.0,
        };
        assert!(t.inverse().is_none());
    }

    #[test]
    fn identity_constant_is_identity() {
        assert_eq!(SimilarityTransform::IDENTITY.apply(3.0, 7.0), (3.0, 7.0));
    }

    #[test]
    fn norm_crop_outputs_112x112() {
        let src = RgbImage::new(640, 480);
        let out = norm_crop(&src, &lms(REF)).unwrap();
        assert_eq!(out.width(), ARCFACE_CROP_SIZE);
        assert_eq!(out.height(), ARCFACE_CROP_SIZE);
    }

    #[test]
    fn norm_crop_rejects_degenerate_landmarks() {
        let src = RgbImage::new(64, 64);
        assert!(norm_crop(&src, &lms([(1.0, 1.0); NUM_LANDMARKS])).is_none());
    }

    #[test]
    fn warp_of_uniform_image_is_uniform() {
        // A constant image must survive any warp unchanged — catches indexing and
        // edge-clamp bugs that would otherwise show up as dark borders.
        let mut src = RgbImage::new(200, 200);
        for b in src.as_mut_slice().iter_mut() {
            *b = 123;
        }
        let out = norm_crop(&src, &lms(REF)).unwrap();
        assert!(
            out.as_slice().iter().all(|&b| b == 123),
            "uniform input must warp to uniform output"
        );
    }

    #[test]
    fn warp_identity_preserves_pixels() {
        // Identity transform on a gradient must reproduce the source region exactly.
        let (w, h) = (32usize, 32usize);
        let mut src = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                src.as_mut_slice()[i] = (x * 8 % 256) as u8;
                src.as_mut_slice()[i + 1] = (y * 8 % 256) as u8;
                src.as_mut_slice()[i + 2] = 7;
            }
        }
        let out = warp_similarity(&src, &SimilarityTransform::IDENTITY, w, h).unwrap();
        assert_eq!(out.as_slice(), src.as_slice());
    }

    #[test]
    fn warp_on_empty_source_returns_none() {
        let src = RgbImage::new(0, 0);
        assert!(warp_similarity(&src, &SimilarityTransform::IDENTITY, 8, 8).is_none());
    }

    #[test]
    fn default_warp_fills_border_black_like_insightface_norm_crop() {
        // 2x2 white source, identity warp into 4x4: in-range pixels stay
        // white, everything beyond the source is borderValue=0 (the OpenCV
        // warpAffine behaviour ArcFace crops were trained with) — never edge
        // replication, which would invent facial texture at crop borders.
        let mut src = RgbImage::new(2, 2);
        for b in src.as_mut_slice().iter_mut() {
            *b = 255;
        }
        let out = warp_similarity(&src, &SimilarityTransform::IDENTITY, 4, 4).unwrap();
        let at = |x: usize, y: usize| {
            let i = (y * 4 + x) * 3;
            (
                out.as_slice()[i],
                out.as_slice()[i + 1],
                out.as_slice()[i + 2],
            )
        };
        assert_eq!(at(0, 0), (255, 255, 255));
        assert_eq!(at(1, 1), (255, 255, 255));
        assert_eq!(at(2, 2), (0, 0, 0), "out-of-range pixel must be black");
        assert_eq!(at(3, 3), (0, 0, 0));
    }

    #[test]
    fn replicate_border_extends_edge_pixels() {
        let mut src = RgbImage::new(2, 2);
        for b in src.as_mut_slice().iter_mut() {
            *b = 200;
        }
        let out = warp_similarity_fill(
            &src,
            &SimilarityTransform::IDENTITY,
            4,
            4,
            BorderMode::Replicate,
        )
        .unwrap();
        let i = (3 * 4 + 3) * 3;
        assert_eq!(
            (
                out.as_slice()[i],
                out.as_slice()[i + 1],
                out.as_slice()[i + 2]
            ),
            (200, 200, 200)
        );
    }

    #[test]
    fn realistic_offcenter_face_lands_on_reference() {
        // Simulate a detector finding a face at 2.4x scale, rotated 12 degrees,
        // translated to (300, 180). Alignment must map it back onto the canonical
        // layout to sub-pixel accuracy; this is the end-to-end contract of the module.
        let theta = 12.0f32.to_radians();
        let (c, s) = (theta.cos(), theta.sin());
        let scale = 2.4f32;
        let mut observed = REF;
        for p in observed.iter_mut() {
            let (x, y) = (p.0 * scale, p.1 * scale);
            *p = (c * x - s * y + 300.0, s * x + c * y + 180.0);
        }
        let t = estimate_arcface_transform(&lms(observed)).unwrap();
        assert!((t.scale() - 1.0 / scale).abs() < 1e-3);
        assert!(
            residual(&t, &observed, &REF) < 1e-2,
            "residual={}",
            residual(&t, &observed, &REF)
        );
    }

    #[test]
    fn landmark_noise_degrades_gracefully() {
        // Real landmarks are noisy. A few px of jitter must still yield a sane
        // transform (scale near 1, small residual) rather than a wild fit.
        let mut noisy = REF;
        let jitter = [
            (1.5f32, -1.0f32),
            (-1.0, 1.5),
            (0.8, 0.9),
            (-1.2, -0.7),
            (1.1, 0.4),
        ];
        for (p, j) in noisy.iter_mut().zip(jitter.iter()) {
            p.0 += j.0;
            p.1 += j.1;
        }
        let t = estimate_similarity(&noisy, &REF).unwrap();
        assert!((t.scale() - 1.0).abs() < 0.1, "scale={}", t.scale());
        assert!(residual(&t, &noisy, &REF) < 3.0);
    }
}
