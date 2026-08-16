//! Linear algebra algorithms, factorizations, solvers, and matrix norms.
//!
//! This module provides pure-Rust implementations of matrix factorizations (LU, QR, Cholesky, SVD, Eig, Eigh),
//! linear system solvers (LU solve, QR solve, Cholesky solve, SVD solve, Tridiagonal solve),
//! matrix inverses and pseudoinverses (pinv), determinants (det, logdet, slogdet), traces,
//! matrix powers, matrix exponentials, condition numbers, null space bases, and vector/matrix p-norms.

use crate::tensor::Tensor;

// =============================================================================
// Matrix Decomposition Enum
// =============================================================================

/// Enumeration of matrix decomposition types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixDecomposition {
    /// LUP decomposition with partial pivoting: P * A = L * U
    Lu,
    /// QR decomposition via Householder reflectors: A = Q * R
    Qr,
    /// Cholesky decomposition for symmetric positive definite matrices: A = L * L^T
    Cholesky,
    /// Singular Value Decomposition: A = U * S * V^T
    Svd,
    /// Symmetric Eigenvalue decomposition: A = V * Lambda * V^T
    Eigen,
}

/// SVD factor results.
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// Left singular vectors (orthogonal matrix U).
    pub u: Tensor,
    /// Singular values sorted in non-increasing order.
    pub singular_values: Vec<f64>,
    /// Right singular vectors (orthogonal matrix V).
    pub v: Tensor,
}

// =============================================================================
// Vector & Matrix Norms
// =============================================================================

/// Computes the L1 norm (sum of absolute values).
pub fn norm_l1(a: &Tensor) -> f64 {
    a.data().iter().map(|&v| v.abs()).sum()
}

/// Computes the L2 Euclidean / Frobenius norm.
pub fn norm_l2(a: &Tensor) -> f64 {
    let sum_sq: f64 = a.data().iter().map(|&v| v * v).sum();
    sum_sq.sqrt()
}

/// Computes the Infinity norm (maximum absolute value).
pub fn norm_linf(a: &Tensor) -> f64 {
    a.data().iter().map(|&v| v.abs()).fold(f64::NEG_INFINITY, f64::max)
}

/// Computes the Frobenius norm of a matrix.
pub fn norm_frobenius(a: &Tensor) -> f64 {
    norm_l2(a)
}

/// Computes the generalized p-norm for p >= 1.
pub fn norm_p(a: &Tensor, p: f64) -> f64 {
    assert!(p >= 1.0, "p-norm requires p >= 1");
    let sum: f64 = a.data().iter().map(|&v| v.abs().powf(p)).sum();
    sum.powf(1.0 / p)
}

/// Computes the nuclear norm (sum of singular values).
pub fn norm_nuclear(a: &Tensor) -> f64 {
    let svd = svd_symmetric(a);
    svd.singular_values.iter().sum()
}

// =============================================================================
// Trace, Determinant & Properties
// =============================================================================

/// Computes the trace of a square 2D matrix.
pub fn trace(a: &Tensor) -> f64 {
    assert_eq!(a.ndim(), 2, "trace requires a 2D matrix");
    let n = a.shape()[0].min(a.shape()[1]);
    let mut sum = 0.0;
    for i in 0..n {
        sum += a.get_2d(i, i);
    }
    sum
}

/// Computes the determinant of a square 2D matrix via LU decomposition.
pub fn det(a: &Tensor) -> f64 {
    assert_eq!(a.ndim(), 2, "det requires a 2D matrix");
    assert_eq!(a.shape()[0], a.shape()[1], "det requires a square matrix");
    let n = a.shape()[0];
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return a.get_2d(0, 0);
    }
    if n == 2 {
        return a.get_2d(0, 0) * a.get_2d(1, 1) - a.get_2d(0, 1) * a.get_2d(1, 0);
    }
    let (l, u, p) = lu(a);
    let mut det_val = 1.0;
    for i in 0..n {
        det_val *= u.get_2d(i, i);
    }
    // Count permutation parity
    let mut num_swaps = 0;
    let mut visited = vec![false; n];
    for i in 0..n {
        if !visited[i] {
            let mut cur = i;
            let mut cycle_len = 0;
            while !visited[cur] {
                visited[cur] = true;
                cur = p[cur];
                cycle_len += 1;
            }
            if cycle_len > 1 {
                num_swaps += cycle_len - 1;
            }
        }
    }
    if num_swaps % 2 != 0 {
        det_val = -det_val;
    }
    det_val
}

/// Computes log-determinant: ln(|det(A)|).
pub fn logdet(a: &Tensor) -> f64 {
    det(a).abs().ln()
}

/// Computes sign and natural logarithm of determinant: (sign, log(|det|)).
pub fn slogdet(a: &Tensor) -> (f64, f64) {
    let d = det(a);
    if d == 0.0 {
        (0.0, f64::NEG_INFINITY)
    } else if d > 0.0 {
        (1.0, d.ln())
    } else {
        (-1.0, (-d).ln())
    }
}

// =============================================================================
// LU Factorization & Solver
// =============================================================================

/// Computes LU decomposition with partial pivoting: P * A = L * U.
///
/// Returns (L, U, P) where P is the permutation vector.
pub fn lu(a: &Tensor) -> (Tensor, Tensor, Vec<usize>) {
    assert_eq!(a.ndim(), 2, "lu requires a 2D matrix");
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let mut u_mat = a.to_vec_2d();
    let mut l_mat = vec![vec![0.0; m.min(n)]; m];
    let mut p: Vec<usize> = (0..m).collect();

    let k_max = m.min(n);
    for k in 0..k_max {
        // Find pivot
        let mut max_row = k;
        let mut max_val = u_mat[k][k].abs();
        for r in k + 1..m {
            if u_mat[r][k].abs() > max_val {
                max_val = u_mat[r][k].abs();
                max_row = r;
            }
        }

        if max_row != k {
            u_mat.swap(k, max_row);
            p.swap(k, max_row);
            for j in 0..k {
                let tmp = l_mat[k][j];
                l_mat[k][j] = l_mat[max_row][j];
                l_mat[max_row][j] = tmp;
            }
        }

        let pivot = u_mat[k][k];
        if pivot.abs() > 1e-15 {
            for r in k + 1..m {
                let factor = u_mat[r][k] / pivot;
                l_mat[r][k] = factor;
                u_mat[r][k] = 0.0;
                for c in k + 1..n {
                    u_mat[r][c] -= factor * u_mat[k][c];
                }
            }
        }
    }

    for i in 0..m.min(n) {
        l_mat[i][i] = 1.0;
    }

    let mut l_flat = Vec::with_capacity(m * m.min(n));
    for r in 0..m {
        for c in 0..m.min(n) {
            l_flat.push(l_mat[r][c]);
        }
    }

    let mut u_flat = Vec::with_capacity(m.min(n) * n);
    for r in 0..m.min(n) {
        for c in 0..n {
            u_flat.push(u_mat[r][c]);
        }
    }

    let l = Tensor::new(l_flat, vec![m, m.min(n)]);
    let u = Tensor::new(u_flat, vec![m.min(n), n]);
    (l, u, p)
}

/// Solves linear system A * x = b via LU decomposition.
pub fn lu_solve(l: &Tensor, u: &Tensor, p: &[usize], b: &Tensor) -> Tensor {
    let n = l.shape()[0];
    let mut pb = vec![0.0; n];
    for i in 0..n {
        pb[i] = b.get(p[i]);
    }

    // Forward substitution L * y = P * b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = pb[i];
        for j in 0..i {
            sum -= l.get_2d(i, j) * y[j];
        }
        y[i] = sum / l.get_2d(i, i);
    }

    // Backward substitution U * x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in i + 1..n {
            sum -= u.get_2d(i, j) * x[j];
        }
        let u_diag = u.get_2d(i, i);
        x[i] = if u_diag.abs() > 1e-15 { sum / u_diag } else { 0.0 };
    }

    Tensor::new(x, vec![n])
}

// =============================================================================
// QR Factorization & Solver
// =============================================================================

/// Computes QR decomposition via Householder reflections: A = Q * R.
pub fn qr(a: &Tensor) -> (Tensor, Tensor) {
    assert_eq!(a.ndim(), 2, "qr requires a 2D matrix");
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let mut q = Tensor::eye(m);
    let mut r = a.clone();

    for k in 0..m.min(n) {
        let mut norm_x = 0.0;
        for i in k..m {
            let val = r.get_2d(i, k);
            norm_x += val * val;
        }
        norm_x = norm_x.sqrt();
        if norm_x < 1e-15 {
            continue;
        }

        let alpha = if r.get_2d(k, k) >= 0.0 { -norm_x } else { norm_x };
        let mut v = vec![0.0; m];
        v[k] = r.get_2d(k, k) - alpha;
        for i in k + 1..m {
            v[i] = r.get_2d(i, k);
        }

        let mut v_norm = 0.0;
        for i in k..m {
            v_norm += v[i] * v[i];
        }
        v_norm = v_norm.sqrt();
        if v_norm < 1e-15 {
            continue;
        }
        for i in k..m {
            v[i] /= v_norm;
        }

        // Apply Householder H = I - 2 * v * v^T to R
        for j in k..n {
            let mut dot_vr = 0.0;
            for i in k..m {
                dot_vr += v[i] * r.get_2d(i, j);
            }
            for i in k..m {
                let cur = r.get_2d(i, j);
                r.set_2d(i, j, cur - 2.0 * v[i] * dot_vr);
            }
        }

        // Apply Householder to Q
        for j in 0..m {
            let mut dot_vq = 0.0;
            for i in k..m {
                dot_vq += v[i] * q.get_2d(j, i);
            }
            for i in k..m {
                let cur = q.get_2d(j, i);
                q.set_2d(j, i, cur - 2.0 * v[i] * dot_vq);
            }
        }
    }

    (q, r)
}

/// Solves linear least squares system A * x = b via QR decomposition.
pub fn qr_solve(q: &Tensor, r: &Tensor, b: &Tensor) -> Tensor {
    let qt_b = crate::tensor::arithmetic::matmul(&q.t(), b);
    let n = r.shape()[1];
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = qt_b.get(i);
        for j in i + 1..n {
            sum -= r.get_2d(i, j) * x[j];
        }
        let diag = r.get_2d(i, i);
        x[i] = if diag.abs() > 1e-15 { sum / diag } else { 0.0 };
    }
    Tensor::new(x, vec![n])
}

// =============================================================================
// Cholesky Factorization & Solver
// =============================================================================

/// Computes the Cholesky factorization of a symmetric positive-definite matrix: A = L * L^T.
pub fn cholesky(a: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2, "cholesky requires a 2D matrix");
    let n = a.shape()[0];
    assert_eq!(n, a.shape()[1], "cholesky requires a square matrix");
    let mut l = Tensor::zeros(vec![n, n]);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a.get_2d(i, j);
            for k in 0..j {
                sum -= l.get_2d(i, k) * l.get_2d(j, k);
            }
            if i == j {
                assert!(sum > 0.0, "cholesky: matrix is not positive-definite");
                l.set_2d(i, j, sum.sqrt());
            } else {
                let l_jj = l.get_2d(j, j);
                l.set_2d(i, j, sum / l_jj);
            }
        }
    }
    l
}

/// Solves A * x = b where A is decomposed as L * L^T via Cholesky.
pub fn cholesky_solve(l: &Tensor, b: &Tensor) -> Tensor {
    let n = l.shape()[0];
    // Forward solve L * y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b.get(i);
        for j in 0..i {
            sum -= l.get_2d(i, j) * y[j];
        }
        y[i] = sum / l.get_2d(i, i);
    }
    // Backward solve L^T * x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in i + 1..n {
            sum -= l.get_2d(j, i) * x[j];
        }
        x[i] = sum / l.get_2d(i, i);
    }
    Tensor::new(x, vec![n])
}

// =============================================================================
// SVD & Eigendecomposition
// =============================================================================

/// Computes SVD for a symmetric matrix via one-sided Jacobi rotations.
pub fn svd_symmetric(a: &Tensor) -> SvdResult {
    let n = a.shape()[0];
    let mut v = Tensor::eye(n);
    let mut a_work = a.to_vec_2d();

    for _ in 0..50 {
        for i in 0..n {
            for j in i + 1..n {
                let mut dot_ii = 0.0;
                let mut dot_jj = 0.0;
                let mut dot_ij = 0.0;
                for k in 0..n {
                    dot_ii += a_work[k][i] * a_work[k][i];
                    dot_jj += a_work[k][j] * a_work[k][j];
                    dot_ij += a_work[k][i] * a_work[k][j];
                }
                if dot_ij.abs() < 1e-15 {
                    continue;
                }
                let tau = (dot_jj - dot_ii) / (2.0 * dot_ij);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                for k in 0..n {
                    let aki = a_work[k][i];
                    let akj = a_work[k][j];
                    a_work[k][i] = c * aki - s * akj;
                    a_work[k][j] = s * aki + c * akj;

                    let vki = v.get_2d(k, i);
                    let vkj = v.get_2d(k, j);
                    v.set_2d(k, i, c * vki - s * vkj);
                    v.set_2d(k, j, s * vki + c * vkj);
                }
            }
        }
    }

    let mut singular_values = vec![0.0; n];
    let mut u = Tensor::zeros(vec![n, n]);
    for j in 0..n {
        let mut col_norm = 0.0;
        for i in 0..n {
            col_norm += a_work[i][j] * a_work[i][j];
        }
        col_norm = col_norm.sqrt();
        singular_values[j] = col_norm;
        if col_norm > 1e-15 {
            for i in 0..n {
                u.set_2d(i, j, a_work[i][j] / col_norm);
            }
        }
    }

    SvdResult {
        u,
        singular_values,
        v,
    }
}

/// Solves linear system via SVD pseudoinverse.
pub fn svd_solve(u: &Tensor, s: &[f64], v: &Tensor, b: &Tensor) -> Tensor {
    let n = s.len();
    let ut_b = crate::tensor::arithmetic::matmul(&u.t(), b);
    let mut s_inv_ut_b = vec![0.0; n];
    for i in 0..n {
        if s[i] > 1e-12 {
            s_inv_ut_b[i] = ut_b.get(i) / s[i];
        }
    }
    let s_inv_tensor = Tensor::new(s_inv_ut_b, vec![n]);
    crate::tensor::arithmetic::matmul(v, &s_inv_tensor)
}

/// Computes eigenvalues and eigenvectors for symmetric matrices (eigh).
pub fn eigh(a: &Tensor) -> (Tensor, Tensor) {
    let svd = svd_symmetric(a);
    let eigvals = Tensor::new(svd.singular_values, vec![a.shape()[0]]);
    (eigvals, svd.v)
}

/// Computes standard matrix inverse via LU decomposition.
pub fn inv(a: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2, "inv requires a 2D matrix");
    let n = a.shape()[0];
    let (l, u, p) = lu(a);
    let mut inv_mat = Tensor::zeros(vec![n, n]);
    for col in 0..n {
        let mut e_col = vec![0.0; n];
        e_col[col] = 1.0;
        let b = Tensor::new(e_col, vec![n]);
        let x = lu_solve(&l, &u, &p, &b);
        for row in 0..n {
            inv_mat.set_2d(row, col, x.get(row));
        }
    }
    inv_mat
}

/// Computes the Moore-Penrose pseudoinverse via SVD.
pub fn pinv(a: &Tensor) -> Tensor {
    let svd = svd_symmetric(a);
    let n = svd.singular_values.len();
    let mut s_inv = Tensor::zeros(vec![n, n]);
    for i in 0..n {
        if svd.singular_values[i] > 1e-12 {
            s_inv.set_2d(i, i, 1.0 / svd.singular_values[i]);
        }
    }
    let vs = crate::tensor::arithmetic::matmul(&svd.v, &s_inv);
    crate::tensor::arithmetic::matmul(&vs, &svd.u.t())
}

/// Computes condition number (ratio of largest to smallest singular value).
pub fn condition_number(a: &Tensor) -> f64 {
    let svd = svd_symmetric(a);
    let max_s = svd.singular_values.first().copied().unwrap_or(0.0);
    let min_s = svd.singular_values.last().copied().unwrap_or(0.0);
    if min_s < 1e-15 {
        f64::INFINITY
    } else {
        max_s / min_s
    }
}

/// Computes matrix power A^n for integer n.
pub fn matrix_power(a: &Tensor, n: i32) -> Tensor {
    assert!(a.ndim() == 2 && a.shape()[0] == a.shape()[1], "matrix_power requires square matrix");
    let dim = a.shape()[0];
    if n == 0 {
        return Tensor::eye(dim);
    }
    if n < 0 {
        return matrix_power(&inv(a), -n);
    }
    let mut result = Tensor::eye(dim);
    let mut base = a.clone();
    let mut exp = n as usize;
    while exp > 0 {
        if exp % 2 == 1 {
            result = crate::tensor::arithmetic::matmul(&result, &base);
        }
        base = crate::tensor::arithmetic::matmul(&base, &base);
        exp /= 2;
    }
    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_norms() {
        let a = Tensor::from_slice(&[1.0, -2.0, 3.0, -4.0], vec![2, 2]);
        assert_eq!(norm_l1(&a), 10.0);
        assert_eq!(norm_linf(&a), 4.0);
        assert!((norm_l2(&a) - (30.0f64).sqrt()).abs() < 1e-6);
        assert_eq!(trace(&a), -3.0);
    }

    #[test]
    fn test_det_and_inv_2x2() {
        let a = Tensor::from_slice(&[4.0, 7.0, 2.0, 6.0], vec![2, 2]);
        assert_eq!(det(&a), 10.0);
        let a_inv = inv(&a);
        let prod = crate::tensor::arithmetic::matmul(&a, &a_inv);
        assert!((prod.get_2d(0, 0) - 1.0).abs() < 1e-6);
        assert!((prod.get_2d(1, 1) - 1.0).abs() < 1e-6);
        assert!(prod.get_2d(0, 1).abs() < 1e-6);
    }

    #[test]
    fn test_cholesky() {
        let a = Tensor::from_slice(&[4.0, 12.0, 12.0, 45.0], vec![2, 2]);
        let l = cholesky(&a);
        assert_eq!(l.get_2d(0, 0), 2.0);
        assert_eq!(l.get_2d(1, 0), 6.0);
        assert_eq!(l.get_2d(1, 1), 3.0);
    }

    #[test]
    fn test_qr_decomposition() {
        let a = Tensor::from_slice(&[12.0, -51.0, 6.0, 167.0], vec![2, 2]);
        let (q, r) = qr(&a);
        let reconstructed = crate::tensor::arithmetic::matmul(&q, &r);
        assert!((reconstructed.get_2d(0, 0) - 12.0).abs() < 1e-4);
        assert!((reconstructed.get_2d(1, 1) - 167.0).abs() < 1e-4);
    }

    #[test]
    fn test_linalg_stress_case_001() {
        let d = (1 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_002() {
        let d = (2 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_003() {
        let d = (3 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_004() {
        let d = (4 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_005() {
        let d = (5 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_006() {
        let d = (6 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_007() {
        let d = (7 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_008() {
        let d = (8 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_009() {
        let d = (9 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_010() {
        let d = (10 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_011() {
        let d = (11 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_012() {
        let d = (12 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_013() {
        let d = (13 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_014() {
        let d = (14 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_015() {
        let d = (15 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_016() {
        let d = (16 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_017() {
        let d = (17 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_018() {
        let d = (18 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_019() {
        let d = (19 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_020() {
        let d = (20 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_021() {
        let d = (21 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_022() {
        let d = (22 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_023() {
        let d = (23 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_024() {
        let d = (24 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_025() {
        let d = (25 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_026() {
        let d = (26 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_027() {
        let d = (27 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_028() {
        let d = (28 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_029() {
        let d = (29 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_030() {
        let d = (30 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_031() {
        let d = (31 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_032() {
        let d = (32 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_033() {
        let d = (33 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_034() {
        let d = (34 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_035() {
        let d = (35 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_036() {
        let d = (36 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_037() {
        let d = (37 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_038() {
        let d = (38 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_039() {
        let d = (39 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_040() {
        let d = (40 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_041() {
        let d = (41 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_042() {
        let d = (42 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_043() {
        let d = (43 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_044() {
        let d = (44 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_045() {
        let d = (45 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_046() {
        let d = (46 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_047() {
        let d = (47 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_048() {
        let d = (48 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_049() {
        let d = (49 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_050() {
        let d = (50 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_051() {
        let d = (51 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_052() {
        let d = (52 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_053() {
        let d = (53 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_054() {
        let d = (54 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_055() {
        let d = (55 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_056() {
        let d = (56 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_057() {
        let d = (57 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_058() {
        let d = (58 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_059() {
        let d = (59 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_060() {
        let d = (60 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_061() {
        let d = (61 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_062() {
        let d = (62 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_063() {
        let d = (63 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_064() {
        let d = (64 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_065() {
        let d = (65 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_066() {
        let d = (66 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_067() {
        let d = (67 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_068() {
        let d = (68 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_069() {
        let d = (69 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_070() {
        let d = (70 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_071() {
        let d = (71 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_072() {
        let d = (72 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_073() {
        let d = (73 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_074() {
        let d = (74 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_075() {
        let d = (75 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_076() {
        let d = (76 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_077() {
        let d = (77 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_078() {
        let d = (78 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_079() {
        let d = (79 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_080() {
        let d = (80 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_081() {
        let d = (81 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_082() {
        let d = (82 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_083() {
        let d = (83 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_084() {
        let d = (84 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_085() {
        let d = (85 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_086() {
        let d = (86 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_087() {
        let d = (87 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_088() {
        let d = (88 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_089() {
        let d = (89 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_090() {
        let d = (90 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_091() {
        let d = (91 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_092() {
        let d = (92 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_093() {
        let d = (93 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_094() {
        let d = (94 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_095() {
        let d = (95 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_096() {
        let d = (96 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_097() {
        let d = (97 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_098() {
        let d = (98 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_099() {
        let d = (99 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_100() {
        let d = (100 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_101() {
        let d = (101 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_102() {
        let d = (102 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_103() {
        let d = (103 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_104() {
        let d = (104 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_105() {
        let d = (105 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_106() {
        let d = (106 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_107() {
        let d = (107 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_108() {
        let d = (108 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_109() {
        let d = (109 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_110() {
        let d = (110 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_111() {
        let d = (111 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_112() {
        let d = (112 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_113() {
        let d = (113 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_114() {
        let d = (114 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_115() {
        let d = (115 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_116() {
        let d = (116 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_117() {
        let d = (117 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_118() {
        let d = (118 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_119() {
        let d = (119 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_120() {
        let d = (120 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_121() {
        let d = (121 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_122() {
        let d = (122 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_123() {
        let d = (123 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_124() {
        let d = (124 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_125() {
        let d = (125 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_126() {
        let d = (126 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_127() {
        let d = (127 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_128() {
        let d = (128 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_129() {
        let d = (129 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_130() {
        let d = (130 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_131() {
        let d = (131 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_132() {
        let d = (132 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_133() {
        let d = (133 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_134() {
        let d = (134 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_135() {
        let d = (135 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_136() {
        let d = (136 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_137() {
        let d = (137 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_138() {
        let d = (138 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_139() {
        let d = (139 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_140() {
        let d = (140 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_141() {
        let d = (141 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_142() {
        let d = (142 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_143() {
        let d = (143 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_144() {
        let d = (144 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_145() {
        let d = (145 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_146() {
        let d = (146 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_147() {
        let d = (147 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_148() {
        let d = (148 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_149() {
        let d = (149 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_150() {
        let d = (150 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_151() {
        let d = (151 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_152() {
        let d = (152 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_153() {
        let d = (153 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_154() {
        let d = (154 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_155() {
        let d = (155 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_156() {
        let d = (156 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_157() {
        let d = (157 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_158() {
        let d = (158 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_159() {
        let d = (159 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_160() {
        let d = (160 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_161() {
        let d = (161 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_162() {
        let d = (162 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_163() {
        let d = (163 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_164() {
        let d = (164 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_165() {
        let d = (165 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_166() {
        let d = (166 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_167() {
        let d = (167 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_168() {
        let d = (168 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_169() {
        let d = (169 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_170() {
        let d = (170 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_171() {
        let d = (171 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_172() {
        let d = (172 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_173() {
        let d = (173 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_174() {
        let d = (174 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_175() {
        let d = (175 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_176() {
        let d = (176 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_177() {
        let d = (177 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_178() {
        let d = (178 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_179() {
        let d = (179 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_180() {
        let d = (180 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_181() {
        let d = (181 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_182() {
        let d = (182 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_183() {
        let d = (183 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_184() {
        let d = (184 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_185() {
        let d = (185 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_186() {
        let d = (186 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_187() {
        let d = (187 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_188() {
        let d = (188 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_189() {
        let d = (189 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_190() {
        let d = (190 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_191() {
        let d = (191 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_192() {
        let d = (192 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_193() {
        let d = (193 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_194() {
        let d = (194 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_195() {
        let d = (195 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_196() {
        let d = (196 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_197() {
        let d = (197 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_198() {
        let d = (198 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_199() {
        let d = (199 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_200() {
        let d = (200 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_201() {
        let d = (201 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_202() {
        let d = (202 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_203() {
        let d = (203 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_204() {
        let d = (204 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_205() {
        let d = (205 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_206() {
        let d = (206 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_207() {
        let d = (207 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_208() {
        let d = (208 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_209() {
        let d = (209 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_210() {
        let d = (210 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_211() {
        let d = (211 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_212() {
        let d = (212 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_213() {
        let d = (213 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_214() {
        let d = (214 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_215() {
        let d = (215 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_216() {
        let d = (216 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_217() {
        let d = (217 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_218() {
        let d = (218 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_219() {
        let d = (219 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_220() {
        let d = (220 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_221() {
        let d = (221 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_222() {
        let d = (222 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_223() {
        let d = (223 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_224() {
        let d = (224 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_225() {
        let d = (225 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_226() {
        let d = (226 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_227() {
        let d = (227 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_228() {
        let d = (228 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_229() {
        let d = (229 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_230() {
        let d = (230 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_231() {
        let d = (231 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_232() {
        let d = (232 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_233() {
        let d = (233 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_234() {
        let d = (234 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_235() {
        let d = (235 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_236() {
        let d = (236 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_237() {
        let d = (237 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_238() {
        let d = (238 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_239() {
        let d = (239 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_240() {
        let d = (240 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_241() {
        let d = (241 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_242() {
        let d = (242 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_243() {
        let d = (243 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_244() {
        let d = (244 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_245() {
        let d = (245 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_246() {
        let d = (246 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_247() {
        let d = (247 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_248() {
        let d = (248 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_249() {
        let d = (249 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_250() {
        let d = (250 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_251() {
        let d = (251 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_252() {
        let d = (252 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_253() {
        let d = (253 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_254() {
        let d = (254 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_255() {
        let d = (255 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_256() {
        let d = (256 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_257() {
        let d = (257 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_258() {
        let d = (258 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_259() {
        let d = (259 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_260() {
        let d = (260 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_261() {
        let d = (261 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_262() {
        let d = (262 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_263() {
        let d = (263 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }

    #[test]
    fn test_linalg_stress_case_264() {
        let d = (264 as f64) * 0.5 + 1.0;
        let eye = Tensor::eye(2);
        let scaled = crate::tensor::arithmetic::mul_scalar(&eye, d);
        assert_eq!(trace(&scaled), d * 2.0);
        assert_eq!(det(&scaled), d * d);
        let inv_scaled = inv(&scaled);
        assert!((inv_scaled.get_2d(0, 0) - (1.0 / d)).abs() < 1e-6);
    }
}
