//! # Advanced Tensor Computation Engine Verification Harness (Stage B, Phases 46-60)
//!
//! Tests advanced linear algebra, 2D spatial convolutions, pooling layers,
//! INT8 quantization, sparse SpMM, padding modes, and statistical quantiles.

use brain_core::tensor::arithmetic as arith;
use brain_core::tensor::conv;
use brain_core::tensor::hist;
use brain_core::tensor::linalg;
use brain_core::tensor::pad;
use brain_core::tensor::pool;
use brain_core::tensor::quant;
use brain_core::tensor::sparse::SparseCOO;
use brain_core::Tensor;

fn approx(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

// -----------------------------------------------------------------------------
// Phase 47: Matrix Inverse & Moore-Penrose Pseudoinverse
// -----------------------------------------------------------------------------
#[test]
fn test_phase47_matrix_inverse_and_pseudoinverse() {
    let a = Tensor::from_slice(&[4.0, 7.0, 2.0, 6.0], vec![2, 2]);
    let a_inv = linalg::inv(&a);
    let identity = arith::matmul(&a, &a_inv);

    assert!(approx(identity.get_2d(0, 0), 1.0, 1e-6));
    assert!(approx(identity.get_2d(1, 1), 1.0, 1e-6));
    assert!(identity.get_2d(0, 1).abs() < 1e-6);
    assert!(identity.get_2d(1, 0).abs() < 1e-6);

    let a_pinv = linalg::pinv(&a);
    let pinv_check = arith::matmul(&arith::matmul(&a, &a_pinv), &a);
    for i in 0..2 {
        for j in 0..2 {
            assert!(approx(pinv_check.get_2d(i, j), a.get_2d(i, j), 1e-5));
        }
    }
}

// -----------------------------------------------------------------------------
// Phase 49 & 50: Matrix Power & Condition Number
// -----------------------------------------------------------------------------
#[test]
fn test_phase49_50_matrix_power_and_condition_number() {
    let a = Tensor::from_slice(&[2.0, 0.0, 0.0, 3.0], vec![2, 2]);
    let a_cubed = linalg::matrix_power(&a, 3);
    assert_eq!(a_cubed.get_2d(0, 0), 8.0);
    assert_eq!(a_cubed.get_2d(1, 1), 27.0);

    let cond = linalg::condition_number(&a);
    assert!(approx(cond, 1.5, 1e-5), "Condition number of diag(2, 3) must be 1.5");
}

// -----------------------------------------------------------------------------
// Phase 51: Vector & Matrix Norms and Trace
// -----------------------------------------------------------------------------
#[test]
fn test_phase51_matrix_norms_and_trace() {
    let a = Tensor::from_slice(&[1.0, -2.0, 3.0, -4.0], vec![2, 2]);
    assert_eq!(linalg::norm_l1(&a), 10.0);
    assert_eq!(linalg::norm_linf(&a), 4.0);
    assert!(approx(linalg::norm_l2(&a), (30.0f64).sqrt(), 1e-6));
    assert_eq!(linalg::trace(&a), -3.0);
}

// -----------------------------------------------------------------------------
// Phase 52: 2D Spatial Convolutions
// -----------------------------------------------------------------------------
#[test]
fn test_phase52_conv2d_sliding_window() {
    // 1 sample, 1 channel, 4x4 input
    let input = Tensor::from_slice(&[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
    ], vec![1, 1, 4, 4]);

    // 1 out_channel, 1 in_channel, 2x2 kernel (all 1s)
    let weight = Tensor::ones(vec![1, 1, 2, 2]);

    let out = conv::conv2d_ext(&input, &weight, None, (1, 1), (0, 0), (1, 1));
    assert_eq!(out.shape(), &[1, 1, 3, 3]);

    // Top-left 2x2 sum: 1 + 2 + 5 + 6 = 14
    assert_eq!(out.get_4d(0, 0, 0, 0), 14.0);
}

// -----------------------------------------------------------------------------
// Phase 53: 2D Spatial Pooling
// -----------------------------------------------------------------------------
#[test]
fn test_phase53_spatial_pooling() {
    let input = Tensor::from_slice(&[
        1.0, 3.0, 2.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ], vec![1, 1, 2, 4]);

    let pooled = pool::max_pool2d(&input, (2, 2), (2, 2), (0, 0));
    assert_eq!(pooled.shape(), &[1, 1, 1, 2]);
    assert_eq!(pooled.get_4d(0, 0, 0, 0), 6.0);
    assert_eq!(pooled.get_4d(0, 0, 0, 1), 8.0);
}

// -----------------------------------------------------------------------------
// Phase 54: INT8 Quantization
// -----------------------------------------------------------------------------
#[test]
fn test_phase54_int8_quantization() {
    let input = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], vec![4]);
    let q_tensor = quant::quantize_per_tensor(&input, 0.1, 0);

    assert_eq!(q_tensor.data(), &[-10, 0, 10, 20]);
    assert_eq!(q_tensor.scale(), 0.1);
    assert_eq!(q_tensor.zero_point(), 0);
}

// -----------------------------------------------------------------------------
// Phase 55: Sparse Matrix Multiplication (SpMM)
// -----------------------------------------------------------------------------
#[test]
fn test_phase55_sparse_coo_spmm() {
    let mut coo = SparseCOO::new((2, 3));
    coo.insert(0, 0, 2.0);
    coo.insert(1, 2, 3.0);

    let dense_b = Tensor::from_slice(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ], vec![3, 2]);

    let spmm_out = coo.spmm(&dense_b);
    assert_eq!(spmm_out.shape(), &[2, 2]);
    // Row 0: 2.0 * [1.0, 2.0] = [2.0, 4.0]
    assert_eq!(spmm_out.get_2d(0, 0), 2.0);
    assert_eq!(spmm_out.get_2d(0, 1), 4.0);
    // Row 1: 3.0 * [5.0, 6.0] = [15.0, 18.0]
    assert_eq!(spmm_out.get_2d(1, 0), 15.0);
    assert_eq!(spmm_out.get_2d(1, 1), 18.0);
}

// -----------------------------------------------------------------------------
// Phase 57: Tensor Padding Modes
// -----------------------------------------------------------------------------
#[test]
fn test_phase57_tensor_padding() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let padded = pad::pad(&t, &[1, 1, 1, 1], "constant", 0.0);
    assert_eq!(padded.shape(), &[4, 4]);
    assert_eq!(padded.get_2d(0, 0), 0.0);
    assert_eq!(padded.get_2d(1, 1), 1.0);
    assert_eq!(padded.get_2d(2, 2), 4.0);
}

// -----------------------------------------------------------------------------
// Phase 59: Histograms & Quantiles
// -----------------------------------------------------------------------------
#[test]
fn test_phase59_histograms_and_quantiles() {
    let data = Tensor::from_slice(&[1.0, 2.0, 2.0, 3.0, 4.0, 5.0], vec![6]);
    let (counts, edges) = hist::histogram(&data, 4, (1.0, 5.0));
    assert_eq!(counts.shape(), &[4]);
    assert_eq!(edges.shape(), &[5]);

    let median = hist::quantile(&data, 0.5);
    assert_eq!(median, 3.0);

    let counts_int = hist::bincount(&data, 6);
    assert_eq!(counts_int.get(2), 2.0); // value 2 appeared twice
}

// -----------------------------------------------------------------------------
// Phase 60: Stage B Master Computation Engine Integration Audit
// -----------------------------------------------------------------------------
#[test]
fn test_phase60_master_computation_engine_audit() {
    // Pipeline: Input -> Conv2D -> MaxPool -> Dense GEMM -> Inversion & Trace
    let x = Tensor::ones(vec![1, 1, 6, 6]);
    let w = Tensor::from_slice(&[0.5; 4], vec![1, 1, 2, 2]);

    let conv_out = conv::conv2d_ext(&x, &w, None, (1, 1), (0, 0), (1, 1));
    assert_eq!(conv_out.shape(), &[1, 1, 5, 5]);

    let pool_out = pool::max_pool2d(&conv_out, (2, 2), (2, 2), (0, 0));
    assert_eq!(pool_out.shape(), &[1, 1, 2, 2]);

    let flat = pool_out.reshape(vec![2, 2]);
    let tr = linalg::trace(&flat);
    assert!(tr > 0.0);
}
