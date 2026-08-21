//! Thread-local memory pool — recycle `Vec<u8>` / image buffers across frames
//! to minimize allocator pressure in the detection hot path.
//!
//! Detection touches the following buffers per frame:
//! - Source frame pixels (grayscale, RGB)
//! - Integral image (`(W+1) × (H+1)` u32)
//! - Pyramid downscaled images (N × smaller allocations)
//! - Detection lists (Vec<Detection>)
//!
//! Without pooling, every call to `IntegralImage::from_gray` and every
//! pyramid level allocates a fresh `Vec<u32>`. With pooling we reuse the
//! same backing storage across frames, keeping steady-state heap usage
//! essentially flat.

use crate::image::{GrayImage, RgbImage};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static POOL: RefCell<Pool> = RefCell::new(Pool::new());
}

struct Pool {
    gray: HashMap<(usize, usize), Vec<GrayImage>>,
    rgb: HashMap<(usize, usize), Vec<RgbImage>>,
    integrals: HashMap<(usize, usize), Vec<Vec<u32>>>,
    integrals_u64: HashMap<(usize, usize), Vec<Vec<u64>>>,
    detections: Vec<Vec<crate::detector::Detection>>,
}

impl Pool {
    fn new() -> Self {
        Self {
            gray: HashMap::new(),
            rgb: HashMap::new(),
            integrals: HashMap::new(),
            integrals_u64: HashMap::new(),
            detections: Vec::new(),
        }
    }
}

/// Acquire a `GrayImage` of the given size from the pool. Returns an owned
/// `GrayImage` that lives until dropped; subsequent requests for the same
/// dimensions may return the same backing allocation if it's been recycled.
pub fn acquire_gray(w: usize, h: usize) -> GrayImage {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        let key = (w, h);
        if let Some(bucket) = p.gray.get_mut(&key) {
            if let Some(mut img) = bucket.pop() {
                img.as_mut_slice().fill(0);
                return img;
            }
        }
        GrayImage::new(w, h)
    })
}

/// Return a `GrayImage` to the pool for reuse on the next request.
pub fn release_gray(img: GrayImage) {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        let key = (img.width(), img.height());
        p.gray.entry(key).or_insert_with(Vec::new).push(img);
    });
}

/// Same as [`acquire_gray`] for RGB.
pub fn acquire_rgb(w: usize, h: usize) -> RgbImage {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        let key = (w, h);
        if let Some(bucket) = p.rgb.get_mut(&key) {
            if let Some(mut img) = bucket.pop() {
                img.as_mut_slice().fill(0);
                return img;
            }
        }
        RgbImage::new(w, h)
    })
}

pub fn release_rgb(img: RgbImage) {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        let key = (img.width(), img.height());
        p.rgb.entry(key).or_insert_with(Vec::new).push(img);
    });
}

/// Acquire a pre-sized `Vec<u32>` for use as an integral image buffer.
/// The buffer is **not** zeroed on recycle — the caller
/// (`IntegralImage::from_gray`) overwrites every cell it reads except the
/// padding row/column, which it zeroes itself, so skip the redundant
/// `memset` over the whole `(W+1)×(H+1)` table on every frame.
pub fn acquire_integral(w: usize, h: usize) -> Vec<u32> {
    let stride = w + 1;
    let needed = stride * (h + 1);
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        let key = (w, h);
        if let Some(bucket) = p.integrals.get_mut(&key) {
            if let Some(mut v) = bucket.pop() {
                if v.capacity() >= needed {
                    v.resize(needed, 0);
                    return v;
                }
            }
        }
        vec![0u32; needed]
    })
}

pub fn release_integral(w: usize, h: usize, buf: Vec<u32>) {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        let key = (w, h);
        p.integrals.entry(key).or_insert_with(Vec::new).push(buf);
    });
}

/// Acquire a pre-sized `Vec<u64>` for the squared integral image
/// (same layout: `(w+1) * (h+1)` elements). Like [`acquire_integral`],
/// recycled buffers are length-reset but not zeroed — the caller zeroes
/// exactly the padding cells it depends on.
pub fn acquire_integral_u64(w: usize, h: usize) -> Vec<u64> {
    let stride = w + 1;
    let needed = stride * (h + 1);
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        let key = (w, h);
        if let Some(bucket) = p.integrals_u64.get_mut(&key) {
            if let Some(mut v) = bucket.pop() {
                if v.capacity() >= needed {
                    v.resize(needed, 0);
                    return v;
                }
            }
        }
        vec![0u64; needed]
    })
}

pub fn release_integral_u64(w: usize, h: usize, buf: Vec<u64>) {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        let key = (w, h);
        p.integrals_u64
            .entry(key)
            .or_insert_with(Vec::new)
            .push(buf);
    });
}

/// Acquire a `Vec<Detection>` that may have spare capacity from a previous frame.
pub fn acquire_detections() -> Vec<crate::detector::Detection> {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        if let Some(mut v) = p.detections.pop() {
            v.clear();
            return v;
        }
        Vec::with_capacity(64)
    })
}

pub fn release_detections(mut v: Vec<crate::detector::Detection>) {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        v.clear();
        if v.capacity() > 4096 {
            return;
        } // don't grow unboundedly
        p.detections.push(v);
    });
}

/// Drop everything in the pool. Useful in tests or shutdown.
pub fn clear() {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        p.gray.clear();
        p.rgb.clear();
        p.integrals.clear();
        p.integrals_u64.clear();
        p.detections.clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_recycles_buffers() {
        clear();
        let img1 = acquire_gray(8, 8);
        let ptr1 = img1.as_slice().as_ptr();
        release_gray(img1);
        let img2 = acquire_gray(8, 8);
        // Same backing buffer (ptr equal) when pool hits.
        assert_eq!(ptr1, img2.as_slice().as_ptr());
        assert_eq!(img2.width(), 8);
        assert_eq!(img2.height(), 8);
    }

    #[test]
    fn pool_resizes_integral() {
        clear();
        let v1 = acquire_integral(10, 10);
        release_integral(10, 10, v1);
        let v2 = acquire_integral(10, 10);
        assert_eq!(v2.len(), 11 * 11);
    }

    #[test]
    fn pool_recycles_integral_u64_zeroed() {
        clear();
        let p1 = {
            let v = acquire_integral_u64(10, 10);
            assert_eq!(v.len(), 11 * 11);
            let p = v.as_ptr();
            release_integral_u64(10, 10, v);
            p
        };
        // Acquire again — the pool must hand back the same allocation with
        // the right length. It is NOT pre-zeroed (the integral builders
        // zero exactly the padding cells they read); length is what the
        // pool must guarantee.
        let mut v = acquire_integral_u64(10, 10);
        assert_eq!(p1, v.as_ptr(), "pool should reuse the backing allocation");
        v.fill(0xDEAD_BEEF);
        release_integral_u64(10, 10, v);
        let v2 = acquire_integral_u64(10, 10);
        assert_eq!(v2.len(), 11 * 11, "recycled buffer must have exact length");
    }
}
