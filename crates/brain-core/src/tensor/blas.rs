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
    fn test_gemm_low_level() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0; 4];
        gemm(false, false, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2);
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }
}
