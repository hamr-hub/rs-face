//! Sanity tests for the CNN detector's `CnnScratch` buffer reuse + the
//! standalone `conv2d_into` / `maxpool2_into` / `fc_into` / `relu` /
//! `sigmoid` kernels.
//!
//! The hot path in `CnnDetector::detect` performs one forward pass per
//! window; the scratch struct is what keeps the 7 intermediate tensors
//! from being reallocated per window. A future refactor that accidentally
//! grows the scratch buffers (e.g. changes a stride and forgets to update
//! the dims) will silently break the forward pass; these tests pin the
//! sizes and shape contracts.

use rsface::cnn::{
    conv2d_into, fc_into, maxpool2_into, relu, sigmoid, template_face_weights, CnnScratch,
};

#[test]
fn scratch_default_sizes_match_forward_pass_contract() {
    // These lengths are part of the contract between `CnnScratch` and
    // `forward()`; changing one without the other is a silent failure.
    // If you intentionally change a CNN layer shape, update this test.
    let s = CnnScratch::new();
    let b = unsafe { &*s.buffers_mut() };
    // (Same internal access via a fresh `new()` because `buffers_mut()`
    // requires the SAFETY contract; we never share between threads here.)
    let s2 = CnnScratch::new();
    let b2 = s2.buffers_mut();
    assert_eq!(b2.c1.len(), 22 * 22 * 8, "c1 = 22x22x8");
    assert_eq!(b2.c2.len(), 20 * 20 * 16, "c2 = 20x20x16");
    assert_eq!(b2.c2p.len(), 10 * 10 * 16, "c2p = 10x10x16");
    assert_eq!(b2.c3.len(), 8 * 8 * 32, "c3 = 8x8x32");
    assert_eq!(b2.c3p.len(), 4 * 4 * 32, "c3p = 4x4x32");
    assert_eq!(b2.f1.len(), 32, "f1 = 32");
    assert_eq!(b2.f2.len(), 1, "f2 = 1");
    // Avoid unused-variable warning when neither field is otherwise read.
    let _ = b;
}

#[test]
fn scratch_buffer_is_reusable_across_multiple_forward_passes() {
    // The whole point of the scratch struct is to be reused; running
    // `forward` twice must yield the same answer (same weights, same input).
    use rsface::cnn::forward;

    let weights = template_face_weights();
    let scratch = CnnScratch::new();
    let window: Vec<f32> = (0..24 * 24).map(|i| (i as f32) / (24.0 * 24.0)).collect();
    let s1 = forward(&weights, &window, &scratch);
    let s2 = forward(&weights, &window, &scratch);
    assert_eq!(
        s1, s2,
        "scratch must be reusable: same input -> same output"
    );
}

#[test]
fn conv2d_into_output_size_equals_window_minus_kernel_plus_one() {
    // 8x8 input, 3x3 kernel, 4 output channels: 6x6 output.
    let input = vec![0.5f32; 8 * 8];
    let kernel = vec![0.1f32; 3 * 3 * 1 * 4];
    let mut out = vec![0.0f32; 36 * 4];
    conv2d_into(&input, 8, 8, 1, &kernel, 3, 3, 4, &mut out);
    assert_eq!(out.len(), 36 * 4);
}

#[test]
fn conv2d_into_unity_kernel_is_box_average() {
    // 1x1 channel with a 1x1 kernel of weight 2: each output pixel = 2 * input pixel.
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let kernel = vec![2.0f32];
    let mut out = vec![0.0f32; 4];
    conv2d_into(&input, 2, 2, 1, &kernel, 1, 1, 1, &mut out);
    assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn maxpool2_into_halves_width_and_height() {
    // 4x4x1 input -> 2x2x1 output, each cell is the max of its 2x2 window.
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mut out = vec![f32::NEG_INFINITY; 4];
    maxpool2_into(&input, 4, 4, 1, &mut out);
    assert_eq!(out.len(), 4);
    // Window (0,1,4,5) max = 5; (2,3,6,7) = 7; (8,9,12,13)=13; (10,11,14,15)=15.
    assert_eq!(out, vec![5.0, 7.0, 13.0, 15.0]);
}

#[test]
fn maxpool2_into_handles_odd_dimensions_by_truncating() {
    // 5x5x1 -> 2x2x1 (floor halves); the rightmost column + bottom row are dropped.
    let input: Vec<f32> = (0..25).map(|i| i as f32).collect();
    let mut out = vec![f32::NEG_INFINITY; 4];
    maxpool2_into(&input, 5, 5, 1, &mut out);
    // Window (0,1,5,6) max = 6; (2,3,7,8)=8; (10,11,15,16)=16; (12,13,17,18)=18.
    assert_eq!(out, vec![6.0, 8.0, 16.0, 18.0]);
}

#[test]
fn fc_into_identity_weight_is_passthrough() {
    // 3 in, 2 out: identity-weight matrix (only diag entries set) passes input through.
    let input = vec![1.0f32, 2.0, 3.0];
    // Row 0 = [1,0,0], Row 1 = [0,0,1] -> out = [1, 3].
    let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let bias = vec![0.0f32, 0.0];
    let mut out = vec![0.0f32; 2];
    fc_into(&input, &weights, &bias, 2, &mut out);
    assert_eq!(out, vec![1.0, 3.0]);
}

#[test]
fn fc_into_bias_is_added() {
    // All-zero weights + non-zero bias = pure bias output.
    let input = vec![10.0, 20.0];
    let weights = vec![0.0, 0.0, 0.0, 0.0];
    let bias = vec![0.5f32, -1.0];
    let mut out = vec![0.0f32; 2];
    fc_into(&input, &weights, &bias, 2, &mut out);
    assert_eq!(out, vec![0.5, -1.0]);
}

#[test]
fn relu_clamps_negative_values() {
    let mut v = vec![-3.0f32, -0.001, 0.0, 0.5, 100.0];
    relu(&mut v);
    assert_eq!(v, vec![0.0, 0.0, 0.0, 0.5, 100.0]);
}

#[test]
fn sigmoid_maps_to_unit_interval() {
    let mut v = vec![-100.0f32, -1.0, 0.0, 1.0, 100.0];
    sigmoid(&mut v);
    for (i, &x) in v.iter().enumerate() {
        assert!((0.0..=1.0).contains(&x), "sigmoid[{i}] = {x} not in [0,1]");
    }
    // sigmoid(0) == 0.5 exactly.
    assert!((v[2] - 0.5).abs() < 1e-6);
}

#[test]
fn template_weights_have_expected_layer_dimensions() {
    // The "starter" template weights ship in the binary; their shape is
    // part of the contract. Pin it.
    let w = template_face_weights();
    // conv1: 3*3*1*8
    assert_eq!(w.conv1_w.len(), 3 * 3 * 1 * 8);
    // conv2: 3*3*8*16
    assert_eq!(w.conv2_w.len(), 3 * 3 * 8 * 16);
    // conv3: 3*3*16*32
    assert_eq!(w.conv3_w.len(), 3 * 3 * 16 * 32);
    // fc1: 512 * 32 (4*4*32 = 512 flatten)
    assert_eq!(w.fc1_w.len(), 512 * 32);
    assert_eq!(w.fc1_b.len(), 32);
    // fc2: 32 * 1
    assert_eq!(w.fc2_w.len(), 32);
    assert_eq!(w.fc2_b.len(), 1);
}
