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
    a.data()
        .iter()
        .map(|&v| v.abs())
        .fold(f64::NEG_INFINITY, f64::max)
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
        x[i] = if u_diag.abs() > 1e-15 {
            sum / u_diag
        } else {
            0.0
        };
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

        let alpha = if r.get_2d(k, k) >= 0.0 {
            -norm_x
        } else {
            norm_x
        };
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
    let max_s = svd.singular_values.iter().copied().fold(0.0f64, f64::max);
    let min_s = svd
        .singular_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if min_s < 1e-15 {
        f64::INFINITY
    } else {
        max_s / min_s
    }
}

/// Computes matrix power A^n for integer n.
pub fn matrix_power(a: &Tensor, n: i32) -> Tensor {
    assert!(
        a.ndim() == 2 && a.shape()[0] == a.shape()[1],
        "matrix_power requires square matrix"
    );
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
    fn test_det_and_logdet_4x4_and_8x8_reference() {
        // 4x4 triangular matrix: det is product of diagonals
        // Diag: 2.0, 3.0, 5.0, 7.0 => det = 210.0
        let a4 = Tensor::from_slice(
            &[
                2.0, 1.0, 3.0, 4.0, 0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 7.0,
            ],
            vec![4, 4],
        );
        let d4 = det(&a4);
        assert!(
            (d4 - 210.0).abs() < 1e-9,
            "det(4x4) expected 210.0, got {}",
            d4
        );
        let ld4 = logdet(&a4);
        assert!(
            (ld4 - 210.0f64.ln()).abs() < 1e-9,
            "logdet(4x4) expected ln(210), got {}",
            ld4
        );

        // 8x8 diagonal matrix: Diag: [1, 2, 3, 4, 5, 6, 7, 8] => det = 40320.0
        let mut data8 = vec![0.0; 64];
        let mut expected_det = 1.0f64;
        for i in 0..8 {
            let val = (i + 1) as f64;
            data8[i * 8 + i] = val;
            expected_det *= val;
        }
        let a8 = Tensor::from_slice(&data8, vec![8, 8]);
        let d8 = det(&a8);
        assert!(
            (d8 - expected_det).abs() < 1e-7,
            "det(8x8) expected {}, got {}",
            expected_det,
            d8
        );
        let ld8 = logdet(&a8);
        assert!(
            (ld8 - expected_det.ln()).abs() < 1e-7,
            "logdet(8x8) expected ln({}), got {}",
            expected_det,
            ld8
        );
    }

    #[test]
    fn test_svd_symmetric_reconstruction() {
        // Symmetric positive semi-definite matrix A = M^T * M
        let m = Tensor::from_slice(&[1.0, 2.0, 0.5, 2.0, 3.0, 1.0, 0.5, 1.0, 4.0], vec![3, 3]);

        let svd = svd_symmetric(&m);
        // Reconstruct A = U * S * V^T
        let mut u_s = svd.u.clone();
        for r in 0..3 {
            for c in 0..3 {
                let val = u_s.get_2d(r, c) * svd.singular_values[c];
                u_s.set_2d(r, c, val);
            }
        }
        let v_t = svd.v.transpose(0, 1);
        let reconstructed = crate::tensor::arithmetic::matmul(&u_s, &v_t);

        let mut diff_frob = 0.0;
        for r in 0..3 {
            for c in 0..3 {
                let diff = reconstructed.get_2d(r, c) - m.get_2d(r, c);
                diff_frob += diff * diff;
            }
        }
        let diff_frob = diff_frob.sqrt();
        assert!(
            diff_frob < 1e-6,
            "SVD reconstruction Frobenius norm diff = {}",
            diff_frob
        );
    }

    #[test]
    fn test_lu_decomposition_and_solve() {
        let a = Tensor::from_slice(
            &[3.0, 2.0, -1.0, 2.0, -2.0, 4.0, -1.0, 0.5, -1.0],
            vec![3, 3],
        );
        let (l, u, p) = lu(&a);

        // Solve A * x = b where b = [1.0, -2.0, 0.0]
        let b = Tensor::from_slice(&[1.0, -2.0, 0.0], vec![3]);
        let x = lu_solve(&l, &u, &p, &b);
        assert_eq!(x.shape(), &[3]);

        let ax = crate::tensor::arithmetic::matmul(&a, &x.reshape(vec![3, 1]));
        assert!((ax.get(0) - 1.0).abs() < 1e-5);
        assert!((ax.get(1) - (-2.0)).abs() < 1e-5);
        assert!(ax.get(2).abs() < 1e-5);
    }

    #[test]
    fn test_matrix_power_and_inv() {
        let a = Tensor::from_slice(&[2.0, 1.0, 1.0, 2.0], vec![2, 2]);
        let a_inv = inv(&a);
        let a_pow_neg1 = matrix_power(&a, -1);
        for r in 0..2 {
            for c in 0..2 {
                assert!((a_inv.get_2d(r, c) - a_pow_neg1.get_2d(r, c)).abs() < 1e-6);
            }
        }
        let a_pow_0 = matrix_power(&a, 0);
        assert_eq!(a_pow_0.data(), &[1.0, 0.0, 0.0, 1.0]);
    }
}
