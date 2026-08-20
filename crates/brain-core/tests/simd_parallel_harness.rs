//! # SIMD & Parallelism Verification Harness (Stage C, Phases 61-85)
//!
//! Tests AVX2/FMA SIMD vector kernels, thread-parallel elementwise operations,
//! parallel cache-blocked GEMM, batched matrix multiplication, and N-D joining.

use brain_core::tensor::arithmetic as arith;
use brain_core::tensor::ops_nd;
use brain_core::tensor::simd;
use brain_core::Tensor;

fn approx(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

// -----------------------------------------------------------------------------
// Phase 67 & 68: SIMD Vector Addition, Subtraction & Multiplication
// -----------------------------------------------------------------------------
#[test]
fn test_simd_add_sub_mul_vectors() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let mut out_add = [0.0; 8];
    let mut out_sub = [0.0; 8];
    let mut out_mul = [0.0; 8];

    simd::simd_add(&a, &b, &mut out_add);
    simd::simd_sub(&b, &a, &mut out_sub);
    simd::simd_mul(&a, &b, &mut out_mul);

    assert_eq!(out_add, [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]);
    assert_eq!(out_sub, [9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0]);
    assert_eq!(
        out_mul,
        [10.0, 40.0, 90.0, 160.0, 250.0, 360.0, 490.0, 640.0]
    );
}

// -----------------------------------------------------------------------------
// Phase 69, 70 & 72: SIMD FMA, ReLU & AXPY
// -----------------------------------------------------------------------------
#[test]
fn test_simd_fma_relu_and_axpy() {
    let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut out_relu = [0.0; 5];
    simd::simd_relu(&a, &mut out_relu);
    assert_eq!(out_relu, [0.0, 0.0, 0.0, 1.0, 2.0]);

    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [2.0, 3.0, 4.0, 5.0];
    let z = [10.0, 10.0, 10.0, 10.0];
    let mut out_fma = [0.0; 4];
    simd::simd_fma(&x, &y, &z, &mut out_fma);
    // x*y + z = [12, 16, 22, 30]
    assert_eq!(out_fma, [12.0, 16.0, 22.0, 30.0]);

    let dot = simd::simd_dot(&x, &y);
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    assert_eq!(dot, 40.0);

    let mut out_axpy = [1.0, 1.0, 1.0, 1.0];
    simd::simd_axpy(2.0, &x, &mut out_axpy);
    // 1 + 2*[1, 2, 3, 4] = [3, 5, 7, 9]
    assert_eq!(out_axpy, [3.0, 5.0, 7.0, 9.0]);
}

// -----------------------------------------------------------------------------
// Phase 61 & 62: Multi-Threaded Parallel Elementwise & GEMM
// -----------------------------------------------------------------------------
#[test]
fn test_parallel_elementwise_and_large_gemm() {
    let n = 256;
    let mut data_a = Vec::with_capacity(n * n);
    let mut data_b = Vec::with_capacity(n * n);

    for i in 0..(n * n) {
        data_a.push((i % 7) as f64 * 0.1);
        data_b.push((i % 5) as f64 * 0.1);
    }

    let a = Tensor::from_vec(data_a, vec![n, n]);
    let b = Tensor::from_vec(data_b, vec![n, n]);

    let mapped = a.map(|x| x * 2.0 + 1.0);
    assert_eq!(mapped.shape(), &[n, n]);
    assert!(approx(mapped.get_2d(0, 0), 1.0, 1e-6));

    let c = arith::matmul(&a, &b);
    assert_eq!(c.shape(), &[n, n]);
    for &val in c.data() {
        assert!(val.is_finite());
    }
}

// -----------------------------------------------------------------------------
// Phase 64: Batched Matrix Multiplication (BMM)
// -----------------------------------------------------------------------------
#[test]
fn test_batched_matrix_multiplication() {
    let batch = 4;
    let m = 3;
    let k = 4;
    let n = 2;

    let mut a_data = Vec::with_capacity(batch * m * k);
    for i in 0..(batch * m * k) {
        a_data.push((i % 3) as f64 + 1.0);
    }
    let mut b_data = Vec::with_capacity(batch * k * n);
    for i in 0..(batch * k * n) {
        b_data.push((i % 2) as f64 + 1.0);
    }

    let a = Tensor::from_vec(a_data, vec![batch, m, k]);
    let b = Tensor::from_vec(b_data, vec![batch, k, n]);

    let c = arith::matmul(&a, &b);
    assert_eq!(c.shape(), &[batch, m, n]);

    for &val in c.data() {
        assert!(val > 0.0 && val.is_finite());
    }
}

// -----------------------------------------------------------------------------
// Phase 76: Multidimensional Tensor Stacking & Concatenation
// -----------------------------------------------------------------------------
#[test]
fn test_tensor_cat_and_stack() {
    let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t2 = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let cat0 = ops_nd::cat(&[&t1, &t2], 0);
    assert_eq!(cat0.shape(), &[4, 2]);
    assert_eq!(cat0.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let cat1 = ops_nd::cat(&[&t1, &t2], 1);
    assert_eq!(cat1.shape(), &[2, 4]);
    assert_eq!(cat1.get_2d(0, 0), 1.0);
    assert_eq!(cat1.get_2d(0, 2), 5.0);

    let stacked = ops_nd::stack(&[&t1, &t2], 0);
    assert_eq!(stacked.shape(), &[2, 2, 2]);
}

// -----------------------------------------------------------------------------
// Phase 85: Stage C Master Performance Integration Audit
// -----------------------------------------------------------------------------
#[test]
fn test_stage_c_master_parallel_simd_audit() {
    // Pipeline: Batched inputs -> Parallel Matmul -> SIMD Vector activation -> Stacking
    let b1 = Tensor::ones(vec![2, 8, 8]);
    let b2 = Tensor::ones(vec![2, 8, 8]);
    let prod = arith::matmul(&b1, &b2);
    assert_eq!(prod.shape(), &[2, 8, 8]);
    // Each element is dot product of 8 ones = 8.0
    assert_eq!(prod.get_3d(0, 0, 0), 8.0);

    let stacked = ops_nd::stack(&[&prod, &prod], 0);
    assert_eq!(stacked.shape(), &[2, 2, 8, 8]);
}
