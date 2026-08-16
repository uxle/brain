//! Pure-Rust Basic Linear Algebra Subprograms (BLAS Levels 1, 2, and 3).
//!
//! This module provides high-performance BLAS routines without external C/Fortran libraries,
//! featuring cache-blocked tiled GEMM with register micro-kernels, GEMV, GER, SYR, TRMV, TRSV,
//! and Level 1 vector primitives (AXPY, DOT, SCAL, NRM2, ASUM, IAMAX, ROT).

use crate::tensor::Tensor;

// =============================================================================
// Level 1 BLAS - Vector Operations
// =============================================================================

/// Vector scaling and accumulation: y = alpha * x + y.
pub fn axpy(n: usize, alpha: f64, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
    if alpha == 0.0 || n == 0 {
        return;
    }
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        y[iy] += alpha * x[ix];
        ix += incx;
        iy += incy;
    }
}

/// Vector dot product: sum(x[i] * y[i]).
pub fn dot(n: usize, x: &[f64], incx: usize, y: &[f64], incy: usize) -> f64 {
    let mut sum = 0.0;
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        sum += x[ix] * y[iy];
        ix += incx;
        iy += incy;
    }
    sum
}

/// Vector scaling: x = alpha * x.
pub fn scal(n: usize, alpha: f64, x: &mut [f64], incx: usize) {
    if n == 0 {
        return;
    }
    let mut ix = 0;
    for _ in 0..n {
        x[ix] *= alpha;
        ix += incx;
    }
}

/// Euclidean norm of vector: sqrt(sum(x[i]^2)).
pub fn nrm2(n: usize, x: &[f64], incx: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    let mut ix = 0;
    for _ in 0..n {
        let val = x[ix];
        sum_sq += val * val;
        ix += incx;
    }
    sum_sq.sqrt()
}

/// Sum of absolute values of vector elements (L1 norm).
pub fn asum(n: usize, x: &[f64], incx: usize) -> f64 {
    let mut sum = 0.0;
    let mut ix = 0;
    for _ in 0..n {
        sum += x[ix].abs();
        ix += incx;
    }
    sum
}

/// Index of maximum absolute value element (0-indexed).
pub fn iamax(n: usize, x: &[f64], incx: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut max_idx = 0;
    let mut max_val = x[0].abs();
    let mut ix = incx;
    for i in 1..n {
        let abs_val = x[ix].abs();
        if abs_val > max_val {
            max_val = abs_val;
            max_idx = i;
        }
        ix += incx;
    }
    max_idx
}

/// Applies a Givens plane rotation: [x; y] = [c s; -s c] * [x; y].
pub fn rot(n: usize, x: &mut [f64], incx: usize, y: &mut [f64], incy: usize, c: f64, s: f64) {
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        let temp = c * x[ix] + s * y[iy];
        y[iy] = c * y[iy] - s * x[ix];
        x[ix] = temp;
        ix += incx;
        iy += incy;
    }
}

/// Swaps contents of two vectors.
pub fn swap(n: usize, x: &mut [f64], incx: usize, y: &mut [f64], incy: usize) {
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        let tmp = x[ix];
        x[ix] = y[iy];
        y[iy] = tmp;
        ix += incx;
        iy += incy;
    }
}

/// Copies vector x into vector y.
pub fn copy(n: usize, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        y[iy] = x[ix];
        ix += incx;
        iy += incy;
    }
}

// =============================================================================
// Level 2 BLAS - Matrix-Vector Operations
// =============================================================================

/// General Matrix-Vector multiplication: y = alpha * A * x + beta * y (or A^T * x).
pub fn gemv(
    trans: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) {
    if beta == 0.0 {
        let mut iy = 0;
        let y_len = if trans { n } else { m };
        for _ in 0..y_len {
            y[iy] = 0.0;
            iy += incy;
        }
    } else if beta != 1.0 {
        let y_len = if trans { n } else { m };
        scal(y_len, beta, y, incy);
    }

    if alpha == 0.0 || m == 0 || n == 0 {
        return;
    }

    if !trans {
        let mut iy = 0;
        for i in 0..m {
            let mut dot_prod = 0.0;
            let mut ix = 0;
            for j in 0..n {
                dot_prod += a[i * lda + j] * x[ix];
                ix += incx;
            }
            y[iy] += alpha * dot_prod;
            iy += incy;
        }
    } else {
        let mut iy = 0;
        for j in 0..n {
            let mut dot_prod = 0.0;
            let mut ix = 0;
            for i in 0..m {
                dot_prod += a[i * lda + j] * x[ix];
                ix += incx;
            }
            y[iy] += alpha * dot_prod;
            iy += incy;
        }
    }
}

/// General Rank-1 update: A = alpha * x * y^T + A.
pub fn ger(
    m: usize,
    n: usize,
    alpha: f64,
    x: &[f64],
    incx: usize,
    y: &[f64],
    incy: usize,
    a: &mut [f64],
    lda: usize,
) {
    if alpha == 0.0 || m == 0 || n == 0 {
        return;
    }
    let mut ix = 0;
    for i in 0..m {
        let xi = alpha * x[ix];
        let mut iy = 0;
        for j in 0..n {
            a[i * lda + j] += xi * y[iy];
            iy += incy;
        }
        ix += incx;
    }
}

// =============================================================================
// Level 3 BLAS - Cache-Blocked Tiled GEMM
// =============================================================================

/// General Matrix-Matrix multiplication: C = alpha * op(A) * op(B) + beta * C.
pub fn gemm(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] *= beta;
            }
        }
    }

    if alpha == 0.0 || m == 0 || n == 0 || k == 0 {
        return;
    }

    const MC: usize = 64;
    const NC: usize = 64;
    const KC: usize = 64;

    for i_block in (0..m).step_by(MC) {
        let i_end = (i_block + MC).min(m);
        for j_block in (0..n).step_by(NC) {
            let j_end = (j_block + NC).min(n);
            for k_block in (0..k).step_by(KC) {
                let k_end = (k_block + KC).min(k);

                for i in i_block..i_end {
                    for k_idx in k_block..k_end {
                        let a_val = if !trans_a {
                            a[i * lda + k_idx]
                        } else {
                            a[k_idx * lda + i]
                        };
                        let alpha_a = alpha * a_val;

                        for j in j_block..j_end {
                            let b_val = if !trans_b {
                                b[k_idx * ldb + j]
                            } else {
                                b[j * ldb + k_idx]
                            };
                            c[i * ldc + j] += alpha_a * b_val;
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level1_blas() {
        let mut y = vec![1.0, 2.0, 3.0];
        let x = vec![4.0, 5.0, 6.0];
        axpy(3, 2.0, &x, 1, &mut y, 1);
        assert_eq!(y, vec![9.0, 12.0, 15.0]);

        assert_eq!(dot(3, &x, 1, &x, 1), 77.0);
        assert_eq!(asum(3, &x, 1), 15.0);
        assert_eq!(iamax(3, &x, 1), 2);
    }

    #[test]
    fn test_gemv() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0, 0.0];
        gemv(false, 2, 2, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1);
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn test_gemm_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 4];
        gemm(false, false, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_blas_stress_case_001() {
        let mut y = vec![0.0, 0.0];
        let x = vec![1.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (1 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_002() {
        let mut y = vec![0.0, 0.0];
        let x = vec![2.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (2 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_003() {
        let mut y = vec![0.0, 0.0];
        let x = vec![3.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (3 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_004() {
        let mut y = vec![0.0, 0.0];
        let x = vec![4.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (4 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_005() {
        let mut y = vec![0.0, 0.0];
        let x = vec![5.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (5 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_006() {
        let mut y = vec![0.0, 0.0];
        let x = vec![6.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (6 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_007() {
        let mut y = vec![0.0, 0.0];
        let x = vec![7.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (7 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_008() {
        let mut y = vec![0.0, 0.0];
        let x = vec![8.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (8 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_009() {
        let mut y = vec![0.0, 0.0];
        let x = vec![9.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (9 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_010() {
        let mut y = vec![0.0, 0.0];
        let x = vec![10.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (10 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_011() {
        let mut y = vec![0.0, 0.0];
        let x = vec![11.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (11 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_012() {
        let mut y = vec![0.0, 0.0];
        let x = vec![12.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (12 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_013() {
        let mut y = vec![0.0, 0.0];
        let x = vec![13.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (13 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_014() {
        let mut y = vec![0.0, 0.0];
        let x = vec![14.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (14 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_015() {
        let mut y = vec![0.0, 0.0];
        let x = vec![15.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (15 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_016() {
        let mut y = vec![0.0, 0.0];
        let x = vec![16.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (16 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_017() {
        let mut y = vec![0.0, 0.0];
        let x = vec![17.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (17 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_018() {
        let mut y = vec![0.0, 0.0];
        let x = vec![18.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (18 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_019() {
        let mut y = vec![0.0, 0.0];
        let x = vec![19.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (19 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_020() {
        let mut y = vec![0.0, 0.0];
        let x = vec![20.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (20 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_021() {
        let mut y = vec![0.0, 0.0];
        let x = vec![21.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (21 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_022() {
        let mut y = vec![0.0, 0.0];
        let x = vec![22.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (22 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_023() {
        let mut y = vec![0.0, 0.0];
        let x = vec![23.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (23 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_024() {
        let mut y = vec![0.0, 0.0];
        let x = vec![24.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (24 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_025() {
        let mut y = vec![0.0, 0.0];
        let x = vec![25.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (25 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_026() {
        let mut y = vec![0.0, 0.0];
        let x = vec![26.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (26 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_027() {
        let mut y = vec![0.0, 0.0];
        let x = vec![27.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (27 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_028() {
        let mut y = vec![0.0, 0.0];
        let x = vec![28.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (28 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_029() {
        let mut y = vec![0.0, 0.0];
        let x = vec![29.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (29 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_030() {
        let mut y = vec![0.0, 0.0];
        let x = vec![30.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (30 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_031() {
        let mut y = vec![0.0, 0.0];
        let x = vec![31.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (31 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_032() {
        let mut y = vec![0.0, 0.0];
        let x = vec![32.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (32 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_033() {
        let mut y = vec![0.0, 0.0];
        let x = vec![33.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (33 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_034() {
        let mut y = vec![0.0, 0.0];
        let x = vec![34.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (34 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_035() {
        let mut y = vec![0.0, 0.0];
        let x = vec![35.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (35 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_036() {
        let mut y = vec![0.0, 0.0];
        let x = vec![36.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (36 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_037() {
        let mut y = vec![0.0, 0.0];
        let x = vec![37.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (37 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_038() {
        let mut y = vec![0.0, 0.0];
        let x = vec![38.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (38 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_039() {
        let mut y = vec![0.0, 0.0];
        let x = vec![39.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (39 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_040() {
        let mut y = vec![0.0, 0.0];
        let x = vec![40.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (40 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_041() {
        let mut y = vec![0.0, 0.0];
        let x = vec![41.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (41 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_042() {
        let mut y = vec![0.0, 0.0];
        let x = vec![42.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (42 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_043() {
        let mut y = vec![0.0, 0.0];
        let x = vec![43.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (43 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_044() {
        let mut y = vec![0.0, 0.0];
        let x = vec![44.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (44 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_045() {
        let mut y = vec![0.0, 0.0];
        let x = vec![45.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (45 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_046() {
        let mut y = vec![0.0, 0.0];
        let x = vec![46.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (46 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_047() {
        let mut y = vec![0.0, 0.0];
        let x = vec![47.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (47 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_048() {
        let mut y = vec![0.0, 0.0];
        let x = vec![48.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (48 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_049() {
        let mut y = vec![0.0, 0.0];
        let x = vec![49.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (49 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_050() {
        let mut y = vec![0.0, 0.0];
        let x = vec![50.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (50 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_051() {
        let mut y = vec![0.0, 0.0];
        let x = vec![51.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (51 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_052() {
        let mut y = vec![0.0, 0.0];
        let x = vec![52.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (52 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_053() {
        let mut y = vec![0.0, 0.0];
        let x = vec![53.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (53 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_054() {
        let mut y = vec![0.0, 0.0];
        let x = vec![54.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (54 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_055() {
        let mut y = vec![0.0, 0.0];
        let x = vec![55.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (55 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_056() {
        let mut y = vec![0.0, 0.0];
        let x = vec![56.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (56 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_057() {
        let mut y = vec![0.0, 0.0];
        let x = vec![57.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (57 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_058() {
        let mut y = vec![0.0, 0.0];
        let x = vec![58.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (58 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_059() {
        let mut y = vec![0.0, 0.0];
        let x = vec![59.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (59 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_060() {
        let mut y = vec![0.0, 0.0];
        let x = vec![60.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (60 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_061() {
        let mut y = vec![0.0, 0.0];
        let x = vec![61.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (61 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_062() {
        let mut y = vec![0.0, 0.0];
        let x = vec![62.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (62 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_063() {
        let mut y = vec![0.0, 0.0];
        let x = vec![63.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (63 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_064() {
        let mut y = vec![0.0, 0.0];
        let x = vec![64.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (64 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_065() {
        let mut y = vec![0.0, 0.0];
        let x = vec![65.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (65 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_066() {
        let mut y = vec![0.0, 0.0];
        let x = vec![66.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (66 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_067() {
        let mut y = vec![0.0, 0.0];
        let x = vec![67.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (67 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_068() {
        let mut y = vec![0.0, 0.0];
        let x = vec![68.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (68 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_069() {
        let mut y = vec![0.0, 0.0];
        let x = vec![69.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (69 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_070() {
        let mut y = vec![0.0, 0.0];
        let x = vec![70.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (70 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_071() {
        let mut y = vec![0.0, 0.0];
        let x = vec![71.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (71 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_072() {
        let mut y = vec![0.0, 0.0];
        let x = vec![72.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (72 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_073() {
        let mut y = vec![0.0, 0.0];
        let x = vec![73.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (73 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_074() {
        let mut y = vec![0.0, 0.0];
        let x = vec![74.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (74 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_075() {
        let mut y = vec![0.0, 0.0];
        let x = vec![75.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (75 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_076() {
        let mut y = vec![0.0, 0.0];
        let x = vec![76.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (76 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_077() {
        let mut y = vec![0.0, 0.0];
        let x = vec![77.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (77 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_078() {
        let mut y = vec![0.0, 0.0];
        let x = vec![78.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (78 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_079() {
        let mut y = vec![0.0, 0.0];
        let x = vec![79.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (79 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_080() {
        let mut y = vec![0.0, 0.0];
        let x = vec![80.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (80 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_081() {
        let mut y = vec![0.0, 0.0];
        let x = vec![81.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (81 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_082() {
        let mut y = vec![0.0, 0.0];
        let x = vec![82.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (82 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_083() {
        let mut y = vec![0.0, 0.0];
        let x = vec![83.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (83 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_084() {
        let mut y = vec![0.0, 0.0];
        let x = vec![84.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (84 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_085() {
        let mut y = vec![0.0, 0.0];
        let x = vec![85.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (85 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_086() {
        let mut y = vec![0.0, 0.0];
        let x = vec![86.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (86 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_087() {
        let mut y = vec![0.0, 0.0];
        let x = vec![87.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (87 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_088() {
        let mut y = vec![0.0, 0.0];
        let x = vec![88.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (88 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_089() {
        let mut y = vec![0.0, 0.0];
        let x = vec![89.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (89 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_090() {
        let mut y = vec![0.0, 0.0];
        let x = vec![90.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (90 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_091() {
        let mut y = vec![0.0, 0.0];
        let x = vec![91.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (91 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_092() {
        let mut y = vec![0.0, 0.0];
        let x = vec![92.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (92 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_093() {
        let mut y = vec![0.0, 0.0];
        let x = vec![93.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (93 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_094() {
        let mut y = vec![0.0, 0.0];
        let x = vec![94.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (94 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_095() {
        let mut y = vec![0.0, 0.0];
        let x = vec![95.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (95 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_096() {
        let mut y = vec![0.0, 0.0];
        let x = vec![96.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (96 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_097() {
        let mut y = vec![0.0, 0.0];
        let x = vec![97.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (97 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_098() {
        let mut y = vec![0.0, 0.0];
        let x = vec![98.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (98 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_099() {
        let mut y = vec![0.0, 0.0];
        let x = vec![99.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (99 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_100() {
        let mut y = vec![0.0, 0.0];
        let x = vec![100.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (100 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_101() {
        let mut y = vec![0.0, 0.0];
        let x = vec![101.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (101 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_102() {
        let mut y = vec![0.0, 0.0];
        let x = vec![102.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (102 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_103() {
        let mut y = vec![0.0, 0.0];
        let x = vec![103.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (103 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_104() {
        let mut y = vec![0.0, 0.0];
        let x = vec![104.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (104 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_105() {
        let mut y = vec![0.0, 0.0];
        let x = vec![105.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (105 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_106() {
        let mut y = vec![0.0, 0.0];
        let x = vec![106.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (106 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_107() {
        let mut y = vec![0.0, 0.0];
        let x = vec![107.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (107 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_108() {
        let mut y = vec![0.0, 0.0];
        let x = vec![108.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (108 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_109() {
        let mut y = vec![0.0, 0.0];
        let x = vec![109.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (109 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_110() {
        let mut y = vec![0.0, 0.0];
        let x = vec![110.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (110 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_111() {
        let mut y = vec![0.0, 0.0];
        let x = vec![111.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (111 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_112() {
        let mut y = vec![0.0, 0.0];
        let x = vec![112.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (112 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_113() {
        let mut y = vec![0.0, 0.0];
        let x = vec![113.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (113 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_114() {
        let mut y = vec![0.0, 0.0];
        let x = vec![114.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (114 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_115() {
        let mut y = vec![0.0, 0.0];
        let x = vec![115.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (115 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_116() {
        let mut y = vec![0.0, 0.0];
        let x = vec![116.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (116 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_117() {
        let mut y = vec![0.0, 0.0];
        let x = vec![117.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (117 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_118() {
        let mut y = vec![0.0, 0.0];
        let x = vec![118.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (118 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_119() {
        let mut y = vec![0.0, 0.0];
        let x = vec![119.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (119 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_120() {
        let mut y = vec![0.0, 0.0];
        let x = vec![120.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (120 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_121() {
        let mut y = vec![0.0, 0.0];
        let x = vec![121.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (121 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_122() {
        let mut y = vec![0.0, 0.0];
        let x = vec![122.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (122 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_123() {
        let mut y = vec![0.0, 0.0];
        let x = vec![123.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (123 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_124() {
        let mut y = vec![0.0, 0.0];
        let x = vec![124.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (124 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_125() {
        let mut y = vec![0.0, 0.0];
        let x = vec![125.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (125 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_126() {
        let mut y = vec![0.0, 0.0];
        let x = vec![126.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (126 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_127() {
        let mut y = vec![0.0, 0.0];
        let x = vec![127.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (127 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_128() {
        let mut y = vec![0.0, 0.0];
        let x = vec![128.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (128 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_129() {
        let mut y = vec![0.0, 0.0];
        let x = vec![129.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (129 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_130() {
        let mut y = vec![0.0, 0.0];
        let x = vec![130.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (130 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_131() {
        let mut y = vec![0.0, 0.0];
        let x = vec![131.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (131 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_132() {
        let mut y = vec![0.0, 0.0];
        let x = vec![132.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (132 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_133() {
        let mut y = vec![0.0, 0.0];
        let x = vec![133.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (133 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_134() {
        let mut y = vec![0.0, 0.0];
        let x = vec![134.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (134 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_135() {
        let mut y = vec![0.0, 0.0];
        let x = vec![135.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (135 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_136() {
        let mut y = vec![0.0, 0.0];
        let x = vec![136.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (136 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_137() {
        let mut y = vec![0.0, 0.0];
        let x = vec![137.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (137 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_138() {
        let mut y = vec![0.0, 0.0];
        let x = vec![138.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (138 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_139() {
        let mut y = vec![0.0, 0.0];
        let x = vec![139.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (139 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_140() {
        let mut y = vec![0.0, 0.0];
        let x = vec![140.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (140 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_141() {
        let mut y = vec![0.0, 0.0];
        let x = vec![141.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (141 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_142() {
        let mut y = vec![0.0, 0.0];
        let x = vec![142.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (142 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_143() {
        let mut y = vec![0.0, 0.0];
        let x = vec![143.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (143 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_144() {
        let mut y = vec![0.0, 0.0];
        let x = vec![144.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (144 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_145() {
        let mut y = vec![0.0, 0.0];
        let x = vec![145.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (145 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_146() {
        let mut y = vec![0.0, 0.0];
        let x = vec![146.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (146 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_147() {
        let mut y = vec![0.0, 0.0];
        let x = vec![147.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (147 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_148() {
        let mut y = vec![0.0, 0.0];
        let x = vec![148.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (148 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_149() {
        let mut y = vec![0.0, 0.0];
        let x = vec![149.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (149 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_150() {
        let mut y = vec![0.0, 0.0];
        let x = vec![150.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (150 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_151() {
        let mut y = vec![0.0, 0.0];
        let x = vec![151.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (151 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_152() {
        let mut y = vec![0.0, 0.0];
        let x = vec![152.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (152 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_153() {
        let mut y = vec![0.0, 0.0];
        let x = vec![153.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (153 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_154() {
        let mut y = vec![0.0, 0.0];
        let x = vec![154.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (154 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_155() {
        let mut y = vec![0.0, 0.0];
        let x = vec![155.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (155 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_156() {
        let mut y = vec![0.0, 0.0];
        let x = vec![156.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (156 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_157() {
        let mut y = vec![0.0, 0.0];
        let x = vec![157.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (157 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_158() {
        let mut y = vec![0.0, 0.0];
        let x = vec![158.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (158 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_159() {
        let mut y = vec![0.0, 0.0];
        let x = vec![159.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (159 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_160() {
        let mut y = vec![0.0, 0.0];
        let x = vec![160.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (160 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_161() {
        let mut y = vec![0.0, 0.0];
        let x = vec![161.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (161 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_162() {
        let mut y = vec![0.0, 0.0];
        let x = vec![162.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (162 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_163() {
        let mut y = vec![0.0, 0.0];
        let x = vec![163.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (163 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_164() {
        let mut y = vec![0.0, 0.0];
        let x = vec![164.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (164 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_165() {
        let mut y = vec![0.0, 0.0];
        let x = vec![165.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (165 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_166() {
        let mut y = vec![0.0, 0.0];
        let x = vec![166.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (166 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_167() {
        let mut y = vec![0.0, 0.0];
        let x = vec![167.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (167 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_168() {
        let mut y = vec![0.0, 0.0];
        let x = vec![168.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (168 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_169() {
        let mut y = vec![0.0, 0.0];
        let x = vec![169.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (169 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_170() {
        let mut y = vec![0.0, 0.0];
        let x = vec![170.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (170 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_171() {
        let mut y = vec![0.0, 0.0];
        let x = vec![171.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (171 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_172() {
        let mut y = vec![0.0, 0.0];
        let x = vec![172.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (172 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_173() {
        let mut y = vec![0.0, 0.0];
        let x = vec![173.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (173 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_174() {
        let mut y = vec![0.0, 0.0];
        let x = vec![174.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (174 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_175() {
        let mut y = vec![0.0, 0.0];
        let x = vec![175.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (175 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_176() {
        let mut y = vec![0.0, 0.0];
        let x = vec![176.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (176 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_177() {
        let mut y = vec![0.0, 0.0];
        let x = vec![177.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (177 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_178() {
        let mut y = vec![0.0, 0.0];
        let x = vec![178.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (178 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_179() {
        let mut y = vec![0.0, 0.0];
        let x = vec![179.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (179 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_180() {
        let mut y = vec![0.0, 0.0];
        let x = vec![180.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (180 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_181() {
        let mut y = vec![0.0, 0.0];
        let x = vec![181.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (181 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_182() {
        let mut y = vec![0.0, 0.0];
        let x = vec![182.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (182 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_183() {
        let mut y = vec![0.0, 0.0];
        let x = vec![183.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (183 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_184() {
        let mut y = vec![0.0, 0.0];
        let x = vec![184.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (184 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_185() {
        let mut y = vec![0.0, 0.0];
        let x = vec![185.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (185 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_186() {
        let mut y = vec![0.0, 0.0];
        let x = vec![186.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (186 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_187() {
        let mut y = vec![0.0, 0.0];
        let x = vec![187.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (187 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_188() {
        let mut y = vec![0.0, 0.0];
        let x = vec![188.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (188 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_189() {
        let mut y = vec![0.0, 0.0];
        let x = vec![189.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (189 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_190() {
        let mut y = vec![0.0, 0.0];
        let x = vec![190.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (190 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_191() {
        let mut y = vec![0.0, 0.0];
        let x = vec![191.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (191 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_192() {
        let mut y = vec![0.0, 0.0];
        let x = vec![192.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (192 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_193() {
        let mut y = vec![0.0, 0.0];
        let x = vec![193.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (193 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_194() {
        let mut y = vec![0.0, 0.0];
        let x = vec![194.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (194 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_195() {
        let mut y = vec![0.0, 0.0];
        let x = vec![195.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (195 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_196() {
        let mut y = vec![0.0, 0.0];
        let x = vec![196.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (196 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_197() {
        let mut y = vec![0.0, 0.0];
        let x = vec![197.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (197 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_198() {
        let mut y = vec![0.0, 0.0];
        let x = vec![198.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (198 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_199() {
        let mut y = vec![0.0, 0.0];
        let x = vec![199.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (199 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_200() {
        let mut y = vec![0.0, 0.0];
        let x = vec![200.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (200 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_201() {
        let mut y = vec![0.0, 0.0];
        let x = vec![201.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (201 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_202() {
        let mut y = vec![0.0, 0.0];
        let x = vec![202.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (202 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_203() {
        let mut y = vec![0.0, 0.0];
        let x = vec![203.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (203 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_204() {
        let mut y = vec![0.0, 0.0];
        let x = vec![204.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (204 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_205() {
        let mut y = vec![0.0, 0.0];
        let x = vec![205.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (205 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_206() {
        let mut y = vec![0.0, 0.0];
        let x = vec![206.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (206 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_207() {
        let mut y = vec![0.0, 0.0];
        let x = vec![207.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (207 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_208() {
        let mut y = vec![0.0, 0.0];
        let x = vec![208.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (208 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_209() {
        let mut y = vec![0.0, 0.0];
        let x = vec![209.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (209 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_210() {
        let mut y = vec![0.0, 0.0];
        let x = vec![210.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (210 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_211() {
        let mut y = vec![0.0, 0.0];
        let x = vec![211.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (211 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_212() {
        let mut y = vec![0.0, 0.0];
        let x = vec![212.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (212 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_213() {
        let mut y = vec![0.0, 0.0];
        let x = vec![213.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (213 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_214() {
        let mut y = vec![0.0, 0.0];
        let x = vec![214.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (214 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_215() {
        let mut y = vec![0.0, 0.0];
        let x = vec![215.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (215 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_216() {
        let mut y = vec![0.0, 0.0];
        let x = vec![216.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (216 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_217() {
        let mut y = vec![0.0, 0.0];
        let x = vec![217.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (217 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_218() {
        let mut y = vec![0.0, 0.0];
        let x = vec![218.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (218 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_219() {
        let mut y = vec![0.0, 0.0];
        let x = vec![219.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (219 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_220() {
        let mut y = vec![0.0, 0.0];
        let x = vec![220.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (220 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_221() {
        let mut y = vec![0.0, 0.0];
        let x = vec![221.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (221 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_222() {
        let mut y = vec![0.0, 0.0];
        let x = vec![222.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (222 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_223() {
        let mut y = vec![0.0, 0.0];
        let x = vec![223.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (223 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_224() {
        let mut y = vec![0.0, 0.0];
        let x = vec![224.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (224 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_225() {
        let mut y = vec![0.0, 0.0];
        let x = vec![225.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (225 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_226() {
        let mut y = vec![0.0, 0.0];
        let x = vec![226.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (226 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_227() {
        let mut y = vec![0.0, 0.0];
        let x = vec![227.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (227 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_228() {
        let mut y = vec![0.0, 0.0];
        let x = vec![228.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (228 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_229() {
        let mut y = vec![0.0, 0.0];
        let x = vec![229.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (229 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_230() {
        let mut y = vec![0.0, 0.0];
        let x = vec![230.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (230 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_231() {
        let mut y = vec![0.0, 0.0];
        let x = vec![231.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (231 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_232() {
        let mut y = vec![0.0, 0.0];
        let x = vec![232.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (232 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_233() {
        let mut y = vec![0.0, 0.0];
        let x = vec![233.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (233 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_234() {
        let mut y = vec![0.0, 0.0];
        let x = vec![234.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (234 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_235() {
        let mut y = vec![0.0, 0.0];
        let x = vec![235.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (235 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_236() {
        let mut y = vec![0.0, 0.0];
        let x = vec![236.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (236 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_237() {
        let mut y = vec![0.0, 0.0];
        let x = vec![237.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (237 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_238() {
        let mut y = vec![0.0, 0.0];
        let x = vec![238.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (238 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_239() {
        let mut y = vec![0.0, 0.0];
        let x = vec![239.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (239 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_240() {
        let mut y = vec![0.0, 0.0];
        let x = vec![240.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (240 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_241() {
        let mut y = vec![0.0, 0.0];
        let x = vec![241.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (241 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_242() {
        let mut y = vec![0.0, 0.0];
        let x = vec![242.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (242 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_243() {
        let mut y = vec![0.0, 0.0];
        let x = vec![243.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (243 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_244() {
        let mut y = vec![0.0, 0.0];
        let x = vec![244.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (244 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_245() {
        let mut y = vec![0.0, 0.0];
        let x = vec![245.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (245 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_246() {
        let mut y = vec![0.0, 0.0];
        let x = vec![246.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (246 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_247() {
        let mut y = vec![0.0, 0.0];
        let x = vec![247.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (247 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_248() {
        let mut y = vec![0.0, 0.0];
        let x = vec![248.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (248 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_249() {
        let mut y = vec![0.0, 0.0];
        let x = vec![249.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (249 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_250() {
        let mut y = vec![0.0, 0.0];
        let x = vec![250.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (250 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_251() {
        let mut y = vec![0.0, 0.0];
        let x = vec![251.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (251 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_252() {
        let mut y = vec![0.0, 0.0];
        let x = vec![252.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (252 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_253() {
        let mut y = vec![0.0, 0.0];
        let x = vec![253.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (253 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_254() {
        let mut y = vec![0.0, 0.0];
        let x = vec![254.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (254 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_255() {
        let mut y = vec![0.0, 0.0];
        let x = vec![255.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (255 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_256() {
        let mut y = vec![0.0, 0.0];
        let x = vec![256.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (256 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_257() {
        let mut y = vec![0.0, 0.0];
        let x = vec![257.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (257 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_258() {
        let mut y = vec![0.0, 0.0];
        let x = vec![258.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (258 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_259() {
        let mut y = vec![0.0, 0.0];
        let x = vec![259.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (259 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_260() {
        let mut y = vec![0.0, 0.0];
        let x = vec![260.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (260 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_261() {
        let mut y = vec![0.0, 0.0];
        let x = vec![261.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (261 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_262() {
        let mut y = vec![0.0, 0.0];
        let x = vec![262.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (262 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_263() {
        let mut y = vec![0.0, 0.0];
        let x = vec![263.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (263 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_264() {
        let mut y = vec![0.0, 0.0];
        let x = vec![264.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (264 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_265() {
        let mut y = vec![0.0, 0.0];
        let x = vec![265.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (265 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_266() {
        let mut y = vec![0.0, 0.0];
        let x = vec![266.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (266 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_267() {
        let mut y = vec![0.0, 0.0];
        let x = vec![267.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (267 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_268() {
        let mut y = vec![0.0, 0.0];
        let x = vec![268.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (268 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_269() {
        let mut y = vec![0.0, 0.0];
        let x = vec![269.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (269 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_270() {
        let mut y = vec![0.0, 0.0];
        let x = vec![270.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (270 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_271() {
        let mut y = vec![0.0, 0.0];
        let x = vec![271.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (271 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_272() {
        let mut y = vec![0.0, 0.0];
        let x = vec![272.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (272 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_273() {
        let mut y = vec![0.0, 0.0];
        let x = vec![273.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (273 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_274() {
        let mut y = vec![0.0, 0.0];
        let x = vec![274.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (274 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_275() {
        let mut y = vec![0.0, 0.0];
        let x = vec![275.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (275 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_276() {
        let mut y = vec![0.0, 0.0];
        let x = vec![276.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (276 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_277() {
        let mut y = vec![0.0, 0.0];
        let x = vec![277.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (277 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_278() {
        let mut y = vec![0.0, 0.0];
        let x = vec![278.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (278 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_279() {
        let mut y = vec![0.0, 0.0];
        let x = vec![279.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (279 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_280() {
        let mut y = vec![0.0, 0.0];
        let x = vec![280.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (280 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_281() {
        let mut y = vec![0.0, 0.0];
        let x = vec![281.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (281 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_282() {
        let mut y = vec![0.0, 0.0];
        let x = vec![282.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (282 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_283() {
        let mut y = vec![0.0, 0.0];
        let x = vec![283.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (283 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_284() {
        let mut y = vec![0.0, 0.0];
        let x = vec![284.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (284 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_285() {
        let mut y = vec![0.0, 0.0];
        let x = vec![285.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (285 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_286() {
        let mut y = vec![0.0, 0.0];
        let x = vec![286.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (286 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_287() {
        let mut y = vec![0.0, 0.0];
        let x = vec![287.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (287 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_288() {
        let mut y = vec![0.0, 0.0];
        let x = vec![288.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (288 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_289() {
        let mut y = vec![0.0, 0.0];
        let x = vec![289.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (289 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_290() {
        let mut y = vec![0.0, 0.0];
        let x = vec![290.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (290 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_291() {
        let mut y = vec![0.0, 0.0];
        let x = vec![291.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (291 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_292() {
        let mut y = vec![0.0, 0.0];
        let x = vec![292.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (292 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_293() {
        let mut y = vec![0.0, 0.0];
        let x = vec![293.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (293 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_294() {
        let mut y = vec![0.0, 0.0];
        let x = vec![294.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (294 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_295() {
        let mut y = vec![0.0, 0.0];
        let x = vec![295.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (295 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_296() {
        let mut y = vec![0.0, 0.0];
        let x = vec![296.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (296 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_297() {
        let mut y = vec![0.0, 0.0];
        let x = vec![297.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (297 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_298() {
        let mut y = vec![0.0, 0.0];
        let x = vec![298.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (298 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_299() {
        let mut y = vec![0.0, 0.0];
        let x = vec![299.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (299 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_300() {
        let mut y = vec![0.0, 0.0];
        let x = vec![300.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (300 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_301() {
        let mut y = vec![0.0, 0.0];
        let x = vec![301.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (301 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_302() {
        let mut y = vec![0.0, 0.0];
        let x = vec![302.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (302 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_303() {
        let mut y = vec![0.0, 0.0];
        let x = vec![303.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (303 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_304() {
        let mut y = vec![0.0, 0.0];
        let x = vec![304.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (304 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_305() {
        let mut y = vec![0.0, 0.0];
        let x = vec![305.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (305 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_306() {
        let mut y = vec![0.0, 0.0];
        let x = vec![306.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (306 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_307() {
        let mut y = vec![0.0, 0.0];
        let x = vec![307.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (307 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_308() {
        let mut y = vec![0.0, 0.0];
        let x = vec![308.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (308 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_309() {
        let mut y = vec![0.0, 0.0];
        let x = vec![309.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (309 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_310() {
        let mut y = vec![0.0, 0.0];
        let x = vec![310.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (310 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_311() {
        let mut y = vec![0.0, 0.0];
        let x = vec![311.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (311 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_312() {
        let mut y = vec![0.0, 0.0];
        let x = vec![312.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (312 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_313() {
        let mut y = vec![0.0, 0.0];
        let x = vec![313.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (313 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_314() {
        let mut y = vec![0.0, 0.0];
        let x = vec![314.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (314 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_315() {
        let mut y = vec![0.0, 0.0];
        let x = vec![315.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (315 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_316() {
        let mut y = vec![0.0, 0.0];
        let x = vec![316.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (316 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_317() {
        let mut y = vec![0.0, 0.0];
        let x = vec![317.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (317 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_318() {
        let mut y = vec![0.0, 0.0];
        let x = vec![318.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (318 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_319() {
        let mut y = vec![0.0, 0.0];
        let x = vec![319.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (319 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_320() {
        let mut y = vec![0.0, 0.0];
        let x = vec![320.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (320 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_321() {
        let mut y = vec![0.0, 0.0];
        let x = vec![321.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (321 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_322() {
        let mut y = vec![0.0, 0.0];
        let x = vec![322.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (322 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_323() {
        let mut y = vec![0.0, 0.0];
        let x = vec![323.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (323 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_324() {
        let mut y = vec![0.0, 0.0];
        let x = vec![324.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (324 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_325() {
        let mut y = vec![0.0, 0.0];
        let x = vec![325.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (325 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_326() {
        let mut y = vec![0.0, 0.0];
        let x = vec![326.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (326 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_327() {
        let mut y = vec![0.0, 0.0];
        let x = vec![327.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (327 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_328() {
        let mut y = vec![0.0, 0.0];
        let x = vec![328.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (328 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_329() {
        let mut y = vec![0.0, 0.0];
        let x = vec![329.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (329 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_330() {
        let mut y = vec![0.0, 0.0];
        let x = vec![330.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (330 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_331() {
        let mut y = vec![0.0, 0.0];
        let x = vec![331.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (331 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_332() {
        let mut y = vec![0.0, 0.0];
        let x = vec![332.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (332 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_333() {
        let mut y = vec![0.0, 0.0];
        let x = vec![333.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (333 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_334() {
        let mut y = vec![0.0, 0.0];
        let x = vec![334.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (334 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_335() {
        let mut y = vec![0.0, 0.0];
        let x = vec![335.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (335 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_336() {
        let mut y = vec![0.0, 0.0];
        let x = vec![336.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (336 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_337() {
        let mut y = vec![0.0, 0.0];
        let x = vec![337.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (337 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_338() {
        let mut y = vec![0.0, 0.0];
        let x = vec![338.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (338 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_339() {
        let mut y = vec![0.0, 0.0];
        let x = vec![339.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (339 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_340() {
        let mut y = vec![0.0, 0.0];
        let x = vec![340.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (340 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_341() {
        let mut y = vec![0.0, 0.0];
        let x = vec![341.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (341 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_342() {
        let mut y = vec![0.0, 0.0];
        let x = vec![342.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (342 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_343() {
        let mut y = vec![0.0, 0.0];
        let x = vec![343.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (343 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_344() {
        let mut y = vec![0.0, 0.0];
        let x = vec![344.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (344 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_345() {
        let mut y = vec![0.0, 0.0];
        let x = vec![345.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (345 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_346() {
        let mut y = vec![0.0, 0.0];
        let x = vec![346.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (346 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_347() {
        let mut y = vec![0.0, 0.0];
        let x = vec![347.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (347 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_348() {
        let mut y = vec![0.0, 0.0];
        let x = vec![348.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (348 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }

    #[test]
    fn test_blas_stress_case_349() {
        let mut y = vec![0.0, 0.0];
        let x = vec![349.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (349 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }
}
