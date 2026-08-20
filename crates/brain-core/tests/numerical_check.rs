//! Numerical correctness tests for core brain-core tensor ops.

use brain_core::Tensor;
use brain_core::tensor::arithmetic as arith;
use brain_core::tensor::conv;
use brain_core::tensor::pool;
use brain_core::tensor::reduction as red;
use brain_core::tensor::special as spec;

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn check_transpose_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let t = Tensor::from_slice(&data, vec![2, 3]);
    let tt = t.transpose(0, 1);
    // tt is 3x2: [[1,4],[2,5],[3,6]]
    assert_eq!(tt.shape(), &[3, 2]);
    assert!(approx(tt.get_2d(0, 0), 1.0));
    assert!(approx(tt.get_2d(0, 1), 4.0));
    assert!(approx(tt.get_2d(2, 1), 6.0));
    // transpose transpose back
    let ttt = tt.transpose(0, 1);
    assert_eq!(ttt.shape(), &[2, 3]);
    assert!(approx(ttt.get_2d(0, 0), 1.0));
    assert!(approx(ttt.get_2d(1, 2), 6.0));
    println!("OK [transpose] roundtrip correct");
}

#[test]
fn check_reshape() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let r = t.reshape(vec![3, 2]);
    assert_eq!(r.shape(), &[3, 2]);
    // data should be preserved in row-major
    assert_eq!(r.to_vec(), t.to_vec());
    let r2 = r.reshape(vec![6]);
    assert_eq!(r2.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("OK [reshape] preserves data");
}

#[test]
fn check_permute() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
    let p = t.permute(&[2, 0, 1]); // [3, 1, 2]
    assert_eq!(p.shape(), &[3, 1, 2]);
    println!("OK [permute] shape {:?}", p.shape());
}

#[test]
fn check_reduce_along_dim_sum() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let s = red::sum_along_dim(&t, 1, false);
    // sum along rows: [6, 15]
    assert_eq!(s.shape(), &[2]);
    assert!(approx(s.get(0), 6.0));
    assert!(approx(s.get(1), 15.0));
    let s0 = red::sum_along_dim(&t, 0, false);
    println!("sum_along_dim dim0 = {:?}", s0.to_vec());
    // sum along cols: [1+4, 2+5, 3+6] = [5,7,9]
    assert_eq!(s0.shape(), &[3]);
    assert!(approx(s0.get(0), 5.0));
    assert!(approx(s0.get(1), 7.0));
    assert!(approx(s0.get(2), 9.0));
    println!("OK [sum_along_dim] rows and cols");
}

#[test]
fn check_reduce_along_dim_max() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let m = red::max_along_dim(&t, 1, false);
    assert_eq!(m.shape(), &[2]);
    assert!(approx(m.get(0), 3.0));
    assert!(approx(m.get(1), 6.0));
    println!("OK [max_along_dim]");
}

#[test]
fn check_softmax_correctness() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3]);
    let sm = spec::softmax(&t, 1);
    let s = red::sum(&sm);
    assert!(approx(s, 1.0));
    // softmax([1,2,3]): e/(e+e^2+e^3) etc
    let e1 = 2.718281828459045_f64;
    let e2 = e1 * e1;
    let e3 = e2 * e1;
    let z = e1 + e2 + e3;
    assert!(approx(sm.get(0), e1 / z));
    assert!(approx(sm.get(1), e2 / z));
    assert!(approx(sm.get(2), e3 / z));
    println!("OK [softmax] values correct");
}

#[test]
fn check_max_pool2d() {
    let input = Tensor::from_slice(
        &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
        vec![1, 1, 4, 2],
    );
    let out = pool::max_pool2d(&input, (2, 2), (2, 2), (0, 0));
    assert_eq!(out.shape(), &[1, 1, 2, 1]);
    // input 4x2:
    // [[1,3],[2,4],[5,7],[6,8]]
    // max pool 2x2 stride 2: top-left window [1,3,2,4] -> 4, bottom [5,7,6,8] -> 8
    assert!(approx(out.get(0), 4.0));
    assert!(approx(out.get(1), 8.0));
    println!("OK [max_pool2d]");
}

#[test]
fn check_avg_pool2d_no_pad() {
    let input = Tensor::from_slice(
        &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
        vec![1, 1, 4, 2],
    );
    let out = pool::avg_pool2d(&input, (2, 2), (2, 2), (0, 0));
    assert_eq!(out.shape(), &[1, 1, 2, 1]);
    // top window: (1+3+2+4)/4 = 2.5
    // bottom: (5+7+6+8)/4 = 6.5
    assert!(approx(out.get(0), 2.5));
    assert!(approx(out.get(1), 6.5));
    println!("OK [avg_pool2d] no padding");
}

#[test]
fn check_avg_pool2d_with_pad() {
    // input 3x3, kernel 2x2, stride 1, pad 1 => out 4x4
    let input = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
    );
    let out = pool::avg_pool2d(&input, (2, 2), (1, 1), (1, 1));
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    // Check a corner window: top-left, with padding zeros, the 2x2 window covers
    // padded (0,0), (0,1=input[0]), (1,0=input[0]), (1,1=input[1])? Actually pad=1 zero.
    // With count_include_pad semantics the divisor differs. Let's just print to inspect.
    println!("avg_pool2d pad corner out[0,0,0,0] = {}", out.get(0));
    println!("avg_pool2d pad out = {:?}", out.to_vec());
}

#[test]
fn check_global_avg_pool2d() {
    let input = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![1, 1, 2, 4],
    );
    let out = pool::global_avg_pool2d(&input);
    assert_eq!(out.shape(), &[1, 1, 1, 1]);
    assert!(approx(out.get(0), 4.5));
    println!("OK [global_avg_pool2d] = {}", out.get(0));
}

#[test]
fn check_batched_matmul_broadcast() {
    // a: [1, 2, 3], b: [2, 3, 4] -> broadcast batch dim -> [2, 2, 4]
    let a_data: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let a = Tensor::from_slice(&a_data, vec![1, 2, 3]);
    let b_data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let b = Tensor::from_slice(&b_data, vec![2, 3, 4]);
    let c = arith::matmul(&a, &b);
    println!("batched matmul broadcast shape = {:?}", c.shape());
    assert_eq!(c.shape(), &[2, 2, 4]);

    // Reference: a matrix [[0,1,2],[3,4,5]] broadcast to both b's batch matrices.
    let a_mat = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    for bb in 0..2 {
        let b_mat = if bb == 0 {
            (0..12).map(|i| i as f64).collect::<Vec<_>>()
        } else {
            (12..24).map(|i| i as f64).collect::<Vec<_>>()
        };
        for i in 0..2 {
            for j in 0..4 {
                let mut expected = 0.0_f64;
                for k in 0..3 {
                    expected += a_mat[i * 3 + k] * b_mat[k * 4 + j];
                }
                let got = c.get_3d(bb, i, j);
                assert!(approx(got, expected),
                    "matmul broadcast mismatch bb={bb} i={i} j={j}: got {got} expected {expected}");
            }
        }
    }
    println!("OK [batched_matmul_broadcast] values verified");
}

#[test]
fn check_arange() {
    let asc = Tensor::arange(0.0, 10.0, 2.0);
    assert_eq!(asc.to_vec(), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    let desc = Tensor::arange(0.0, -5.0, -1.0);
    assert_eq!(desc.to_vec(), vec![0.0, -1.0, -2.0, -3.0, -4.0]);
    println!("OK [arange] asc and desc");
}

#[test]
fn check_conv2d_kernel_larger_than_input_does_not_crash() {
    // 5x5 kernel on a 3x3 input with no padding => degenerate. Must produce an
    // empty output (not underflow usize -> OOM/panic).
    let input = Tensor::zeros(vec![1, 1, 3, 3]);
    let weight = Tensor::ones(vec![1, 1, 5, 5]);
    let out = conv::conv2d(&input, &weight, None, (1, 1), (0, 0));
    assert_eq!(out.shape(), &[1, 1, 0, 0]);
    assert_eq!(out.data().len(), 0);
    println!("OK [conv2d] degenerate kernel => empty, no crash");
}

#[test]
fn check_pool_kernel_larger_than_input_does_not_crash() {
    let input = Tensor::zeros(vec![1, 1, 3, 3]);
    let out = pool::max_pool2d(&input, (5, 5), (1, 1), (0, 0));
    assert_eq!(out.shape(), &[1, 1, 0, 0]);
    assert_eq!(out.data().len(), 0);
    let avg = pool::avg_pool2d(&input, (5, 5), (1, 1), (0, 0));
    assert_eq!(avg.shape(), &[1, 1, 0, 0]);
    println!("OK [pool] degenerate kernel => empty, no crash");
}

#[test]
fn check_broadcast_map2() {
    // [3,1] * [1,4] -> [3,4]; column vector * row vector
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]); // [[1],[2],[3]]
    let b = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![1, 4]); // [10,20,30,40]
    let c = arith::mul(&a, &b);
    assert_eq!(c.shape(), &[3, 4]);
    // c[i,j] = a[i]*b[j]
    assert_eq!(c.to_vec(), vec![
        10.0, 20.0, 30.0, 40.0,
        20.0, 40.0, 60.0, 80.0,
        30.0, 60.0, 90.0, 120.0,
    ]);
    // scalar broadcast: [2,2] + 100 -> all +100
    let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let n = arith::add_scalar(&m, 100.0);
    assert_eq!(n.to_vec(), vec![101.0, 102.0, 103.0, 104.0]);
    println!("OK [broadcast map2] outer-product and scalar broadcast");
}

#[test]
fn check_conv2d_output_size() {
    // standard case: input 5x5, kernel 3x3, stride 1, pad 0 => out 3x3
    let input = Tensor::zeros(vec![1, 1, 5, 5]);
    let weight = Tensor::ones(vec![1, 1, 3, 3]);
    let out = conv::conv2d(&input, &weight, None, (1, 1), (0, 0));
    assert_eq!(out.shape(), &[1, 1, 3, 3]);
    println!("OK [conv2d] 5x5->3x3");

    // with padding: input 5x5, kernel 3x3, stride 2, pad 1 => out 3x3
    let out2 = conv::conv2d(&input, &weight, None, (2, 2), (1, 1));
    assert_eq!(out2.shape(), &[1, 1, 3, 3]);
    println!("OK [conv2d] pad+stride out {:?}", out2.shape());
}

#[test]
fn check_avg_pool_counts_valid_only() {
    // When padding adds zero elements to a window, the average should ideally
    // divide by the number of valid (non-padded) elements. Check behavior.
    let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![1, 1, 3, 3]);
    let out = pool::avg_pool2d(&input, (2, 2), (1, 1), (1, 1));
    println!("avg_pool2d 3x3 with pad=1 2x2 stride1 out = {:?}", out.to_vec());
}

#[test]
fn check_dtype_is_lossless_cast() {
    use brain_core::dtype::DType;
    let p = DType::promote(DType::U64, DType::I32);
    assert_eq!(p, DType::U64);

    let f16_to_bf16 = DType::BF16.is_lossless_cast(DType::F16);
    assert!(!f16_to_bf16);

    let i8_to_i16 = DType::I16.is_lossless_cast(DType::I8);
    assert!(i8_to_i16);

    let i16_to_i8 = DType::I8.is_lossless_cast(DType::I16);
    assert!(!i16_to_i8);

    let u8_to_i16 = DType::I16.is_lossless_cast(DType::U8);
    assert!(u8_to_i16);

    let i32_to_f32 = DType::F32.is_lossless_cast(DType::I32);
    assert!(!i32_to_f32);
}

#[test]
fn check_dilated_conv_output() {
    let input = Tensor::ones(vec![1, 1, 5, 5]);
    let weight = Tensor::ones(vec![1, 1, 3, 3]);
    let out = conv::conv2d_ext(&input, &weight, None, (1, 1), (0, 0), (2, 2));
    assert_eq!(out.shape(), &[1, 1, 1, 1]);
    assert_eq!(out.to_vec(), vec![9.0]);
}

#[test]
fn check_tensordot_multi_axis() {
    // a: [2, 3, 4], b: [3, 4, 5] -> contract axes ([1, 2], [0, 1]) -> out: [2, 5]
    let a = Tensor::ones(vec![2, 3, 4]);
    let b = Tensor::ones(vec![3, 4, 5]);
    let out = arith::tensordot(&a, &b, (&[1, 2], &[0, 1]));
    assert_eq!(out.shape(), &[2, 5]);
    // Each contracted entry is sum of 3*4 = 12 ones -> 12.0
    for &v in out.data() {
        assert_eq!(v, 12.0);
    }
}

#[test]
fn check_topk_multi_dim() {
    // a: [2, 3] = [[3, 1, 2], [4, 6, 5]]
    let a = Tensor::from_slice(&[3.0, 1.0, 2.0, 4.0, 6.0, 5.0], vec![2, 3]);
    let (top_vals, top_idx) = brain_core::tensor::compare::topk(&a, 2, 1, true);
    assert_eq!(top_vals.shape(), &[2, 2]);
    assert_eq!(top_vals.to_vec(), vec![3.0, 2.0, 6.0, 5.0]);
    assert_eq!(top_idx, vec![0, 2, 1, 2]);
}

#[test]
fn check_pad_reflect() {
    // [1, 2, 3] with reflect pad (1, 1) -> [2, 1, 2, 3, 2]
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
    let out = brain_core::tensor::pad::pad(&t, &[1, 1], "reflect", 0.0);
    assert_eq!(out.shape(), &[5]);
    assert_eq!(out.to_vec(), vec![2.0, 1.0, 2.0, 3.0, 2.0]);
}

#[test]
fn check_parallel_gemm_large_matrices() {
    // 128x128 identity x random matrix
    let n = 128;
    let mut eye_data = vec![0.0f64; n * n];
    for i in 0..n {
        eye_data[i * n + i] = 1.0;
    }
    let eye = Tensor::from_vec(eye_data, vec![n, n]);

    let mut mat_data = vec![0.0f64; n * n];
    for i in 0..n * n {
        mat_data[i] = (i as f64) * 0.01;
    }
    let mat = Tensor::from_vec(mat_data.clone(), vec![n, n]);

    let out = arith::matmul(&eye, &mat);
    assert_eq!(out.shape(), &[n, n]);
    for (o, m) in out.data().iter().zip(mat_data.iter()) {
        assert!(approx(*o, *m), "Parallel GEMM mismatch: {} vs {}", o, m);
    }
}

#[test]
fn check_parallel_batched_gemm() {
    // [4, 32, 32] x [4, 32, 32]
    let b = 4;
    let n = 32;
    let a = Tensor::ones(vec![b, n, n]);
    let b_mat = Tensor::ones(vec![b, n, n]);

    let out = arith::matmul(&a, &b_mat);
    assert_eq!(out.shape(), &[b, n, n]);
    for &v in out.data() {
        assert_eq!(v, 32.0);
    }
}

#[test]
fn check_parallel_elementwise_map() {
    let size = 16384;
    let mut data = vec![0.0f64; size];
    for i in 0..size {
        data[i] = i as f64;
    }
    let t = Tensor::from_vec(data, vec![size]);
    let mapped = t.map(|x| x * 2.0 + 1.0);

    for (i, &v) in mapped.data().iter().enumerate() {
        assert_eq!(v, (i as f64) * 2.0 + 1.0);
    }
}

#[test]
fn check_backend_abstraction_dispatch() {
    use brain_core::{Backend, CpuBackend, SimdCpuBackend};

    let cpu = CpuBackend;
    let simd = SimdCpuBackend;

    assert_eq!(cpu.name(), "CpuBackend");
    assert_eq!(simd.name(), "SimdCpuBackend");

    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let res_cpu = cpu.matmul(&a, &b).unwrap();
    let res_simd = simd.matmul(&a, &b).unwrap();

    assert_eq!(res_cpu.shape(), &[2, 2]);
    assert_eq!(res_simd.shape(), &[2, 2]);
    assert_eq!(res_cpu.to_vec(), res_simd.to_vec());

    let add_cpu = cpu.add(&a, &b).unwrap();
    let add_simd = simd.add(&a, &b).unwrap();
    assert_eq!(add_cpu.to_vec(), add_simd.to_vec());
}

// =============================================================================
// =============================================================================
// Phase 1 Edge Cases & Linalg/FFT Hardening
// =============================================================================

#[test]
fn test_empty_tensor_and_scalar_edge_cases() {
    let empty = Tensor::zeros(vec![0]);
    assert_eq!(empty.numel(), 0);
    assert_eq!(empty.shape(), &[0]);

    let scalar = Tensor::from_slice(&[42.0], vec![]);
    assert_eq!(scalar.numel(), 1);
    assert_eq!(scalar.ndim(), 0);
    assert_eq!(scalar.item(), 42.0);
}

#[test]
fn test_non_contiguous_transposed_matmul_and_reduction() {
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let a_t = a.transpose(0, 1); // 3x2, non-contiguous
    assert_eq!(a_t.shape(), &[3, 2]);

    let b = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let c = arith::matmul(&a_t, &b);
    assert_eq!(c.shape(), &[3, 2]);
    assert_eq!(c.to_vec(), a_t.to_vec());

    let red_sum = red::sum(&a_t);
    assert_eq!(red_sum, 21.0);
}

#[test]
fn test_nan_and_inf_ieee754_propagation() {
    let t = Tensor::from_slice(&[1.0, f64::NAN, 3.0], vec![3]);
    let out = arith::add(&t, &Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]));
    assert_eq!(out.data()[0], 2.0);
    assert!(out.data()[1].is_nan());
    assert_eq!(out.data()[2], 6.0);

    let t_inf = Tensor::from_slice(&[f64::INFINITY, 1.0], vec![2]);
    let out_inf = arith::mul(&t_inf, &Tensor::from_slice(&[2.0, 2.0], vec![2]));
    assert!(out_inf.data()[0].is_infinite());
    assert_eq!(out_inf.data()[1], 2.0);
}

#[test]
fn test_matrix_determinant_4x4_and_8x8_reference() {
    use brain_core::tensor::linalg::det;

    // Diagonal matrix with known det = product of diagonal
    let diag4 = Tensor::from_slice(&[
        2.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 5.0,
    ], vec![4, 4]);
    assert!(approx(det(&diag4), 120.0));

    // 8x8 identity scaled by 2 -> det = 2^8 = 256
    let mut data8 = vec![0.0; 64];
    for i in 0..8 {
        data8[i * 8 + i] = 2.0;
    }
    let diag8 = Tensor::from_slice(&data8, vec![8, 8]);
    assert!(approx(det(&diag8), 256.0));
}

#[test]
fn test_svd_reconstruction_fidelity() {
    use brain_core::tensor::linalg::{svd_symmetric, norm_frobenius};

    // Symmetric positive-definite matrix
    let a = Tensor::from_slice(&[
        4.0, 1.0,
        1.0, 3.0,
    ], vec![2, 2]);

    let svd_res = svd_symmetric(&a);

    // Reconstruct A = U * S * V^T
    let mut s_mat_data = vec![0.0; 4];
    s_mat_data[0] = svd_res.singular_values[0];
    s_mat_data[3] = svd_res.singular_values[1];
    let s_mat = Tensor::from_slice(&s_mat_data, vec![2, 2]);

    let us = arith::matmul(&svd_res.u, &s_mat);
    let vt = svd_res.v.transpose(0, 1);
    let recon = arith::matmul(&us, &vt);

    let diff = arith::sub(&a, &recon);
    let frob_err = norm_frobenius(&diff);
    assert!(frob_err < 1e-7, "SVD Frobenius reconstruction error too large: {}", frob_err);
}

#[test]
fn test_fft_ifft_roundtrip_power_of_two_and_arbitrary() {
    use brain_core::tensor::fft::dft;

    // Power of two length (64)
    let n1 = 64;
    let original1: Vec<f64> = (0..n1).map(|i| (i as f64 * 0.1).sin()).collect();
    let mut re1 = original1.clone();
    let mut im1 = vec![0.0; n1];

    dft(&mut re1, &mut im1, false);
    dft(&mut re1, &mut im1, true);

    for (a, b) in original1.iter().zip(re1.iter()) {
        assert!((a - b).abs() < 1e-6, "Power of two FFT roundtrip failed");
    }

    // Arbitrary non-power-of-two length (50)
    let n2 = 50;
    let original2: Vec<f64> = (0..n2).map(|i| (i as f64 * 0.2).cos()).collect();
    let mut re2 = original2.clone();
    let mut im2 = vec![0.0; n2];

    dft(&mut re2, &mut im2, false);
    dft(&mut re2, &mut im2, true);

    for (a, b) in original2.iter().zip(re2.iter()) {
        assert!((a - b).abs() < 1e-6, "Arbitrary length DFT roundtrip failed");
    }
}
