//! # Core Tensor & Computation Engine Verification Harness (Stage B, Phases 35-45)
//!
//! Tests low-level mathematical kernels, memory allocators, shape broadcasting,
//! reduction algebra, GEMM cache blocking, and linear algebra decompositions.

use brain_core::dtype::DType;
use brain_core::memory::{is_aligned, PAGE_SIZE, CACHE_LINE_SIZE};
use brain_core::random::{BrainRng, PCG32, Rng, NormalDist};
use brain_core::serialization::{TensorArchive, Crc32};
use brain_core::shape::broadcast_shapes;
use brain_core::tensor::arithmetic as arith;
use brain_core::tensor::fft;
use brain_core::tensor::linalg;
use brain_core::tensor::reduction as red;
use brain_core::Tensor;

fn approx(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

// -----------------------------------------------------------------------------
// Phase 35: DType Representation & Promotion
// -----------------------------------------------------------------------------
#[test]
fn test_phase35_dtype_properties_and_promotion_lattice() {
    assert_eq!(DType::F64.size_bytes(), 8);
    assert_eq!(DType::F32.size_bytes(), 4);
    assert_eq!(DType::I64.size_bytes(), 8);
    assert_eq!(DType::I32.size_bytes(), 4);
    assert_eq!(DType::Bool.size_bytes(), 1);

    assert!(DType::F32.is_float());
    assert!(DType::I64.is_int());

    assert_eq!(DType::promote(DType::F32, DType::F64), DType::F64);
    assert_eq!(DType::promote(DType::I32, DType::F32), DType::F32);
    assert_eq!(DType::promote(DType::Bool, DType::I64), DType::I64);
}

// -----------------------------------------------------------------------------
// Phase 36: Aligned Memory Buffers & Alignments
// -----------------------------------------------------------------------------
#[test]
fn test_phase36_memory_alignment_invariants() {
    let raw_buf = vec![0u8; 128];
    let ptr = raw_buf.as_ptr();
    assert!(is_aligned(ptr, 1)); // byte aligned always
    assert_eq!(PAGE_SIZE, 4096);
    assert_eq!(CACHE_LINE_SIZE, 64);
}

// -----------------------------------------------------------------------------
// Phase 37: Indexing, Slicing & Views
// -----------------------------------------------------------------------------
#[test]
fn test_phase37_tensor_strided_views_and_indexing() {
    let t = Tensor::from_slice(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ], vec![3, 3]);

    assert_eq!(t.get_2d(1, 2), 6.0);
    assert_eq!(t.get_2d(2, 0), 7.0);

    let transposed = t.t();
    assert_eq!(transposed.get_2d(2, 1), 6.0);
    assert_eq!(transposed.get_2d(0, 2), 7.0);

    let contig = transposed.contiguous();
    assert_eq!(contig.shape(), &[3, 3]);
    assert_eq!(contig.get_2d(2, 1), 6.0);
}

// -----------------------------------------------------------------------------
// Phase 38: Deterministic PRNG Engines & Distributions
// -----------------------------------------------------------------------------
#[test]
fn test_phase38_deterministic_prng_sampling() {
    let mut rng1 = BrainRng::new(42, 84);
    let mut rng2 = BrainRng::new(42, 84);

    let sample1 = rng1.next_f64();
    let sample2 = rng2.next_f64();
    assert_eq!(sample1, sample2, "PRNG must be bit-exact reproducible from identical seed");

    let mut pcg = PCG32::new(12345, 54321);
    let normal = NormalDist::new(0.0, 1.0);
    let mut sum = 0.0;
    let n = 1000;
    for _ in 0..n {
        sum += normal.sample(&mut pcg);
    }
    let mean = sum / (n as f64);
    assert!(mean.abs() < 0.15, "Standard normal sample mean should be near 0: got {}", mean);
}

// -----------------------------------------------------------------------------
// Phase 39: Binary v2 Serialization & CRC32
// -----------------------------------------------------------------------------
#[test]
fn test_phase39_crc32_and_tensor_archive() {
    let data = b"Brain Neural Architecture Core";
    let crc = Crc32::compute(data);
    assert_ne!(crc, 0);

    let mut archive = TensorArchive::new();
    let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    archive.add("layer1.weight", t1.clone());

    let bytes = archive.serialize().expect("Archive to bytes");
    assert!(!bytes.is_empty());

    let loaded = TensorArchive::deserialize(&bytes).expect("Archive from bytes");
    let loaded_t1 = loaded.get("layer1.weight").expect("Tensor lookup");
    assert_eq!(loaded_t1.shape(), t1.shape());
    assert_eq!(loaded_t1.to_vec(), t1.to_vec());
}

// -----------------------------------------------------------------------------
// Phase 40: Multidimensional Shape Algebra & Broadcast
// -----------------------------------------------------------------------------
#[test]
fn test_phase40_broadcast_shape_algebra() {
    let b = broadcast_shapes(&[2, 1, 4], &[1, 3, 4]);
    assert_eq!(b, vec![2, 3, 4]);

    let a = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
    let b_t = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![1, 3]);
    let out = &a + &b_t;
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.get_2d(0, 0), 11.0);
    assert_eq!(out.get_2d(0, 2), 31.0);
    assert_eq!(out.get_2d(1, 1), 22.0);
}

// -----------------------------------------------------------------------------
// Phase 41: Arithmetic Operations
// -----------------------------------------------------------------------------
#[test]
fn test_phase41_tensor_arithmetic_ops() {
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_slice(&[10.0, 10.0, 10.0, 10.0], vec![2, 2]);
    let sum = &a + &b;
    assert_eq!(sum.to_vec(), vec![11.0, 12.0, 13.0, 14.0]);

    let scaled = sum.map(|x| x * 2.0);
    assert_eq!(scaled.to_vec(), vec![22.0, 24.0, 26.0, 28.0]);
}

// -----------------------------------------------------------------------------
// Phase 42: Reductions (Sum, Mean, Var, Std, Min, Max)
// -----------------------------------------------------------------------------
#[test]
fn test_phase42_reduction_algebra() {
    let t = Tensor::from_slice(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ], vec![2, 3]);

    let sum_0 = red::sum_along_dim(&t, 0, false);
    assert_eq!(sum_0.to_vec(), vec![5.0, 7.0, 9.0]);

    let mean_1 = red::mean_along_dim(&t, 1, false);
    assert_eq!(mean_1.to_vec(), vec![2.0, 5.0]);

    let total_sum = red::sum(&t);
    assert_eq!(total_sum, 21.0);

    let total_mean = red::mean(&t);
    assert_eq!(total_mean, 3.5);

    let min_v = red::min(&t);
    let max_v = red::max(&t);
    assert_eq!(min_v, 1.0);
    assert_eq!(max_v, 6.0);
}

// -----------------------------------------------------------------------------
// Phase 43: BLAS GEMM Cache Blocking
// -----------------------------------------------------------------------------
#[test]
fn test_phase43_cache_blocked_gemm() {
    let m = 16;
    let k = 32;
    let n = 24;

    let mut a_data = Vec::with_capacity(m * k);
    for i in 0..(m * k) {
        a_data.push((i % 7) as f64 * 0.5);
    }
    let mut b_data = Vec::with_capacity(k * n);
    for i in 0..(k * n) {
        b_data.push((i % 5) as f64 * 0.25);
    }

    let a = Tensor::from_vec(a_data, vec![m, k]);
    let b = Tensor::from_vec(b_data, vec![k, n]);

    let c = arith::matmul(&a, &b);
    assert_eq!(c.shape(), &[m, n]);

    // Check single element mathematically
    let mut expected_0_0 = 0.0;
    for idx in 0..k {
        expected_0_0 += a.get_2d(0, idx) * b.get_2d(idx, 0);
    }
    assert!(approx(c.get_2d(0, 0), expected_0_0, 1e-6));
}

// -----------------------------------------------------------------------------
// Phase 44: Fast Fourier Transform (FFT & IFFT)
// -----------------------------------------------------------------------------
#[test]
fn test_phase44_fft_and_ifft_exact_reconstruction() {
    let orig_re = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let orig_im = vec![0.0; 8];

    let mut re = orig_re.clone();
    let mut im = orig_im.clone();

    // Forward FFT
    fft::fft_radix2(&mut re, &mut im, false);

    // Inverse IFFT
    fft::fft_radix2(&mut re, &mut im, true);

    for i in 0..8 {
        assert!(approx(re[i], orig_re[i], 1e-6), "Re diff at {}: got {}, expected {}", i, re[i], orig_re[i]);
        assert!(approx(im[i], 0.0, 1e-6), "Im diff at {}: got {}", i, im[i]);
    }
}

// -----------------------------------------------------------------------------
// Phase 45: Linear Algebra (LU, QR, Cholesky, SVD, Solvers)
// -----------------------------------------------------------------------------
#[test]
fn test_phase45_linear_algebra_factorizations_and_solvers() {
    // 1. Cholesky: A = L * L^T on Symmetric Positive Definite matrix
    // A = [[4, 2], [2, 5]] -> L = [[2, 0], [1, 2]]
    let a_spd = Tensor::from_slice(&[4.0, 2.0, 2.0, 5.0], vec![2, 2]);
    let l = linalg::cholesky(&a_spd);
    assert_eq!(l.shape(), &[2, 2]);
    assert!(approx(l.get_2d(0, 0), 2.0, 1e-6));
    assert!(approx(l.get_2d(0, 1), 0.0, 1e-6));
    assert!(approx(l.get_2d(1, 0), 1.0, 1e-6));
    assert!(approx(l.get_2d(1, 1), 2.0, 1e-6));

    // 2. QR Decomposition: A = Q * R
    let (q, r) = linalg::qr(&a_spd);
    let qr_recon = arith::matmul(&q, &r);
    for i in 0..2 {
        for j in 0..2 {
            assert!(approx(qr_recon.get_2d(i, j), a_spd.get_2d(i, j), 1e-6));
        }
    }

    // 3. Linear Solve: A * x = b
    let b = Tensor::from_slice(&[8.0, 14.0], vec![2]);
    let (l_lu, u_lu, p) = linalg::lu(&a_spd);
    let x = linalg::lu_solve(&l_lu, &u_lu, &p, &b);
    assert_eq!(x.shape(), &[2]);
    // 4*x0 + 2*x1 = 8, 2*x0 + 5*x1 = 14 => x0 = 0.75, x1 = 2.5
    assert!(approx(x.get(0), 0.75, 1e-5));
    assert!(approx(x.get(1), 2.50, 1e-5));

    // 4. Determinant
    let d = linalg::det(&a_spd); // 4*5 - 2*2 = 16
    assert!(approx(d, 16.0, 1e-6));
}
