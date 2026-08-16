//! Linear algebra operations for tensors in the Brain deep learning framework.
//!
//! This module provides pure-Rust implementations of common linear algebra
//! operations including norms, determinant, inverse, decomposition methods,
//! and matrix utilities. No external dependencies (LAPACK, BLAS) are used.
//!
//! # Available Operations
//!
//! * **Norms**: L1, L2, Linf, Frobenius, matrix norms
//! * **Matrix properties**: trace, rank, condition number, determinant
//! * **Decompositions**: LU, QR, Cholesky, SVD (symmetric), eigendecomposition
//! * **Solvers**: matrix inverse, linear system solve (Ax=b)
//! * **Matrix functions**: matrix power, matrix exponential, matrix square root

use crate::tensor::Tensor;

// =============================================================================
// Norm Operations
// =============================================================================

/// Computes the L1 norm (sum of absolute values).
pub fn norm_l1(a: &Tensor) -> f64 {
    a.data().iter().map(|&v| v.abs()).sum()
}

/// Computes the L2 norm (Euclidean norm / Frobenius norm for matrices).
pub fn norm_l2(a: &Tensor) -> f64 {
    let sum_sq: f64 = a.data().iter().map(|&v| v * v).sum();
    sum_sq.sqrt()
}

/// Computes the Linf norm (maximum absolute value).
pub fn norm_linf(a: &Tensor) -> f64 {
    a.data().iter().map(|&v| v.abs()).fold(f64::NEG_INFINITY, f64::max)
}

/// Computes the Frobenius norm of a matrix.
pub fn norm_frobenius(a: &Tensor) -> f64 {
    norm_l2(a)
}

/// Computes the p-norm for scalar p >= 1.
pub fn norm_p(a: &Tensor, p: f64) -> f64 {
    assert!(p >= 1.0, "p-norm requires p >= 1");
    let sum: f64 = a.data().iter().map(|&v| v.abs().powf(p)).sum();
    sum.powf(1.0 / p)
}

/// Computes the nuclear norm (sum of singular values) for a symmetric matrix.
pub fn norm_nuclear(a: &Tensor) -> f64 {
    let svd = svd_symmetric(a);
    svd.singular_values.iter().sum()
}

// =============================================================================
// Trace and Diagonal
// =============================================================================

/// Computes the trace of a square matrix (sum of diagonal elements).
pub fn trace(a: &Tensor) -> f64 {
    assert!(a.is_matrix(), "Trace requires a 2D matrix");
    let (rows, cols) = (a.shape()[0], a.shape()[1]);
    let n = rows.min(cols);
    let mut sum = 0.0;
    for i in 0..n {
        sum += a.get_index(&[i, i]);
    }
    sum
}

/// Extracts the diagonal of a matrix as a 1D tensor.
pub fn diag(a: &Tensor) -> Tensor {
    assert!(a.is_matrix(), "Diag requires a 2D matrix");
    let (rows, cols) = (a.shape()[0], a.shape()[1]);
    let n = rows.min(cols);
    let data: Vec<f64> = (0..n).map(|i| a.get_index(&[i, i])).collect();
    Tensor::new(data, vec![n])
}

// =============================================================================
// Determinant
// =============================================================================

/// Computes the determinant of a square matrix.
pub fn det(a: &Tensor) -> f64 {
    assert!(a.is_matrix(), "Determinant requires a 2D matrix");
    let n = a.shape()[0];
    assert_eq!(n, a.shape()[1], "Matrix must be square");

    match n {
        0 => 1.0,
        1 => a.get(0),
        2 => a.get(0) * a.get(3) - a.get(1) * a.get(2),
        3 => det_3x3(a),
        _ => {
            // Use LU decomposition for larger matrices
            let lu = lu_decompose(a);
            let mut d = lu.determinant_sign;
            for i in 0..n { d *= lu.u.get_index(&[i, i]); }
            d
        }
    }
}

/// 3x3 determinant using cofactor expansion.
fn det_3x3(a: &Tensor) -> f64 {
    let m = a.data();
    m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

// =============================================================================
// Matrix Inverse (Gauss-Jordan Elimination)
// =============================================================================

/// Computes the inverse of a square matrix using Gauss-Jordan elimination.
pub fn inv(a: &Tensor) -> Tensor {
    assert!(a.is_matrix(), "Inverse requires a 2D matrix");
    let n = a.shape()[0];
    assert_eq!(n, a.shape()[1], "Matrix must be square");

    // Create augmented matrix [A | I]
    let mut m = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            m[i * 2 * n + j] = a.get_index(&[i, j]);
            m[i * 2 * n + n + i] = 1.0;
        }
    }

    // Gauss-Jordan elimination
    for col in 0..n {
        // Find pivot
        let mut pivot_row = col;
        let mut max_val = m[col * 2 * n + col].abs();
        for row in (col + 1)..n {
            let val = m[row * 2 * n + col].abs();
            if val > max_val {
                max_val = val;
                pivot_row = row;
            }
        }

        if max_val < 1e-12 {
            panic!("Matrix is singular (column {} has no pivot)", col);
        }

        // Swap rows
        if pivot_row != col {
            for j in 0..2 * n {
                m.swap(col * 2 * n + j, pivot_row * 2 * n + j);
            }
        }

        // Scale pivot row
        let pivot = m[col * 2 * n + col];
        for j in 0..2 * n {
            m[col * 2 * n + j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = m[row * 2 * n + col];
                for j in 0..2 * n {
                    m[row * 2 * n + j] -= factor * m[col * 2 * n + j];
                }
            }
        }
    }

    // Extract inverse
    let mut inv_data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv_data[i * n + j] = m[i * 2 * n + n + j];
        }
    }

    Tensor::new(inv_data, vec![n, n])
}

// =============================================================================
// Linear System Solver (Ax = b)
// =============================================================================

/// Solves the linear system Ax = b for a square matrix A.
pub fn solve(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.is_matrix(), "A must be a 2D matrix");
    let n = a.shape()[0];
    assert_eq!(n, a.shape()[1], "A must be square");
    assert!(b.is_vector() || b.is_matrix(), "b must be 1D or 2D");

    let nrhs = if b.ndim() == 1 { 1 } else { b.shape()[1] };

    // Create augmented system [A | b]
    let mut m = vec![0.0; n * (n + nrhs)];
    for i in 0..n {
        for j in 0..n {
            m[i * (n + nrhs) + j] = a.get_index(&[i, j]);
        }
        for j in 0..nrhs {
            let val = if b.ndim() == 1 { b.get(i) } else { b.get_index(&[i, j]) };
            m[i * (n + nrhs) + n + j] = val;
        }
    }

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = m[col * (n + nrhs) + col].abs();
        for row in (col + 1)..n {
            let val = m[row * (n + nrhs) + col].abs();
            if val > max_val { max_val = val; pivot_row = row; }
        }
        if max_val < 1e-12 { panic!("Matrix is singular"); }
        if pivot_row != col {
            for j in 0..(n + nrhs) { m.swap(col * (n + nrhs) + j, pivot_row * (n + nrhs) + j); }
        }
        let pivot = m[col * (n + nrhs) + col];
        for j in col..(n + nrhs) { m[col * (n + nrhs) + j] /= pivot; }
        for row in (col + 1)..n {
            let factor = m[row * (n + nrhs) + col];
            for j in col..(n + nrhs) {
                m[row * (n + nrhs) + j] -= factor * m[col * (n + nrhs) + j];
            }
        }
    }

    // Back substitution
    let mut x_data = vec![0.0; n * nrhs];
    for j in 0..nrhs {
        for i in (0..n).rev() {
            let mut sum = m[i * (n + nrhs) + n + j];
            for k in (i + 1)..n {
                sum -= m[i * (n + nrhs) + k] * x_data[k * nrhs + j];
            }
            x_data[i * nrhs + j] = sum;
        }
    }

    if nrhs == 1 {
        Tensor::new(x_data, vec![n])
    } else {
        Tensor::new(x_data, vec![n, nrhs])
    }
}

// =============================================================================
// LU Decomposition
// =============================================================================

/// Result of LU decomposition.
pub struct LUDecomposition {
    /// Lower triangular matrix L.
    pub l: Tensor,
    /// Upper triangular matrix U.
    pub u: Tensor,
    /// Permutation vector.
    pub piv: Vec<usize>,
    /// Sign of the determinant (based on number of row swaps).
    pub determinant_sign: f64,
}

/// LU decomposition with partial pivoting: PA = LU.
pub fn lu_decompose(a: &Tensor) -> LUDecomposition {
    let n = a.shape()[0];
    let mut u = a.data().to_vec();
    let mut l = vec![0.0; n * n];
    let mut piv: Vec<usize> = (0..n).collect();
    let mut sign = 1.0;

    for i in 0..n {
        l[i * n + i] = 1.0;
    }

    for k in 0..n {
        // Find pivot
        let mut max_val = u[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            if u[i * n + k].abs() > max_val {
                max_val = u[i * n + k].abs();
                max_row = i;
            }
        }
        if max_row != k {
            piv.swap(k, max_row);
            u.swap(k * n..(k + 1) * n, max_row * n..(max_row + 1) * n);
            if k > 0 {
                l.swap(k * n..k * n + k, max_row * n..max_row * n + k);
            }
            sign = -sign;
        }
        if u[k * n + k].abs() < 1e-12 { continue; }

        for i in (k + 1)..n {
            l[i * n + k] = u[i * n + k] / u[k * n + k];
            for j in k..n {
                u[i * n + j] -= l[i * n + k] * u[k * n + j];
            }
        }
    }

    LUDecomposition {
        l: Tensor::new(l, vec![n, n]),
        u: Tensor::new(u, vec![n, n]),
        piv,
        determinant_sign: sign,
    }
}

// =============================================================================
// QR Decomposition (Householder Reflections)
// =============================================================================

/// Result of QR decomposition.
pub struct QRDecomposition {
    /// Orthogonal matrix Q.
    pub q: Tensor,
    /// Upper triangular matrix R.
    pub r: Tensor,
}

/// QR decomposition using Householder reflections: A = QR.
pub fn qr_decompose(a: &Tensor) -> QRDecomposition {
    let m = a.shape()[0];
    let n = a.shape()[1];
    let mut r = a.data().to_vec();
    let mut q = identity_matrix(m);

    for k in 0..n.min(m - 1) {
        // Compute Householder vector for column k
        let mut x = vec![0.0; m - k];
        for i in 0..(m - k) { x[i] = r[(k + i) * n + k]; }

        let norm_x: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm_x < 1e-12 { continue; }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * norm_x;

        let norm_v: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm_v < 1e-12 { continue; }
        for v in x.iter_mut() { *v /= norm_v; }

        // Apply Householder reflection: R = R - 2 * v * (v^T * R)
        for j in k..n {
            let dot: f64 = (0..(m - k)).map(|i| x[i] * r[(k + i) * n + j]).sum();
            for i in 0..(m - k) {
                r[(k + i) * n + j] -= 2.0 * x[i] * dot;
            }
        }

        // Apply to Q: Q = Q - 2 * (Q * v) * v^T
        for i in 0..m {
            let dot: f64 = (0..(m - k)).map(|j| x[j] * q[i * m + k + j]).sum();
            for j in 0..(m - k) {
                q[i * m + k + j] -= 2.0 * dot * x[j];
            }
        }
    }

    QRDecomposition {
        q: Tensor::new(q, vec![m, m]),
        r: Tensor::new(r, vec![m, n]),
    }
}

fn identity_matrix(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n { m[i * n + i] = 1.0; }
    m
}

// =============================================================================
// Cholesky Decomposition
// =============================================================================

/// Cholesky decomposition: A = L * L^T where L is lower triangular.
/// A must be symmetric positive definite.
pub fn cholesky(a: &Tensor) -> Tensor {
    assert!(a.is_matrix(), "Cholesky requires 2D matrix");
    let n = a.shape()[0];
    assert_eq!(n, a.shape()[1], "Matrix must be square");

    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a.get_index(&[i, j]);
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 { panic!("Matrix is not positive definite at ({}, {})", i, j); }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Tensor::new(l, vec![n, n])
}

// =============================================================================
// SVD for Symmetric Matrices (Jacobi Eigenvalue Algorithm)
// =============================================================================

/// Result of SVD decomposition.
pub struct SVDResult {
    /// Left singular vectors (columns).
    pub u: Tensor,
    /// Singular values in descending order.
    pub singular_values: Vec<f64>,
    /// Right singular vectors (columns).
    pub v: Tensor,
}

/// SVD for symmetric matrices using the Jacobi eigenvalue algorithm.
pub fn svd_symmetric(a: &Tensor) -> SVDResult {
    assert!(a.is_matrix(), "SVD requires 2D matrix");
    let n = a.shape()[0];

    // Symmetrize
    let mut m = vec![0.0; n * n];
    for i in 0..n { for j in 0..n { m[i * n + j] = (a.get_index(&[i, j]) + a.get_index(&[j, i])) / 2.0; } }

    let (eigenvalues, eigenvectors) = jacobi_eigen(&m, n);

    // Sort by absolute eigenvalue (descending)
    let mut pairs: Vec<(f64, Vec<f64>)> = eigenvalues.iter().zip(eigenvectors.chunks(n)).map(|(&v, e)| (v.abs(), e.to_vec())).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut s = Vec::with_capacity(n);
    let mut u_data = vec![0.0; n * n];
    for (i, (val, vec)) in pairs.iter().enumerate() {
        s.push(*val);
        for j in 0..n { u_data[j * n + i] = vec[j]; }
    }

    SVDResult {
        u: Tensor::new(u_data.clone(), vec![n, n]),
        singular_values: s,
        v: Tensor::new(u_data, vec![n, n]),
    }
}

/// Jacobi eigenvalue algorithm for symmetric matrices.
fn jacobi_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = identity_matrix(n);
    let max_iter = 100 * n * n;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut p = 0; let mut q = 1;
        let mut max_val = (a[p * n + q]).abs();
        for i in 0..n {
            for j in (i + 1)..n {
                if (a[i * n + j]).abs() > max_val {
                    max_val = (a[i * n + j]).abs();
                    p = i; q = j;
                }
            }
        }
        if max_val < 1e-12 { break; }

        // Compute rotation
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let theta;
        if (app - aqq).abs() < 1e-12 {
            theta = std::f64::consts::PI / 4.0;
        } else {
            theta = 0.5 * ((app - aqq) / (2.0 * apq)).atan();
        }
        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation
        for r in 0..n {
            if r != p && r != q {
                let arp = a[r * n + p];
                let arq = a[r * n + q];
                a[r * n + p] = c * arp + s * arq;
                a[p * n + r] = a[r * n + p];
                a[r * n + q] = -s * arp + c * arq;
                a[q * n + r] = a[r * n + q];
            }
        }

        a[p * n + p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        a[q * n + q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // Update eigenvectors
        for r in 0..n {
            let vrp = v[r * n + p];
            let vrq = v[r * n + q];
            v[r * n + p] = c * vrp + s * vrq;
            v[r * n + q] = -s * vrp + c * vrq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

// =============================================================================
// Eigenvalue Decomposition for Symmetric Matrices
// =============================================================================

/// Result of eigendecomposition.
pub struct EighResult {
    /// Eigenvalues.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as columns.
    pub eigenvectors: Tensor,
}

/// Eigenvalue decomposition for symmetric matrices.
pub fn eigh(a: &Tensor) -> EighResult {
    assert!(a.is_matrix(), "Eigh requires 2D matrix");
    let n = a.shape()[0];
    let mut m = vec![0.0; n * n];
    for i in 0..n { for j in 0..n { m[i * n + j] = (a.get_index(&[i, j]) + a.get_index(&[j, i])) / 2.0; } }

    let (eigenvalues, eigenvectors) = jacobi_eigen(&m, n);

    // Sort eigenvalues (ascending)
    let mut pairs: Vec<(f64, Vec<f64>)> = eigenvalues.iter().zip(eigenvectors.chunks(n))
        .map(|(&v, e)| (v, e.to_vec())).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut sorted_vals = Vec::with_capacity(n);
    let mut sorted_vecs = vec![0.0; n * n];
    for (i, (val, vec)) in pairs.iter().enumerate() {
        sorted_vals.push(*val);
        for j in 0..n { sorted_vecs[j * n + i] = vec[j]; }
    }

    EighResult {
        eigenvalues: sorted_vals,
        eigenvectors: Tensor::new(sorted_vecs, vec![n, n]),
    }
}

// =============================================================================
// Matrix Power and Matrix Functions
// =============================================================================

/// Computes A^n for integer n using repeated squaring.
pub fn matrix_power(a: &Tensor, n: i32) -> Tensor {
    assert!(a.is_matrix(), "Matrix power requires 2D matrix");
    let size = a.shape()[0];
    assert_eq!(size, a.shape()[1], "Matrix must be square");

    if n == 0 { return Tensor::identity(size); }
    if n < 0 { return inv(&matrix_power(a, -n)); }
    if n == 1 { return a.clone(); }

    let mut result = Tensor::identity(size);
    let mut base = a.clone();
    let mut exp = n;

    while exp > 0 {
        if exp % 2 == 1 {
            result = crate::tensor::arithmetic::matmul(&result, &base);
        }
        base = crate::tensor::arithmetic::matmul(&base, &base);
        exp /= 2;
    }

    result
}

/// Matrix exponential using Padé approximation.
pub fn matrix_exp(a: &Tensor) -> Tensor {
    assert!(a.is_matrix(), "Matrix exp requires 2D matrix");
    let n = a.shape()[0];

    // Scale down for numerical stability
    let norm = norm_linf(a);
    let s = if norm > 1.0 { (norm.log2().ceil() as usize).max(1) } else { 1 };
    let scale = 1.0 / (1 << s) as f64;

    let scaled = crate::tensor::arithmetic::mul_scalar(a, scale);

    // Padé approximation (order 13)
    let a_sq = crate::tensor::arithmetic::matmul(&scaled, &scaled);
    let a4 = crate::tensor::arithmetic::matmul(&a_sq, &a_sq);
    let a6 = crate::tensor::arithmetic::matmul(&a_sq, &a4);

    // Numerator coefficients (Padé [6,6])
    let u = crate::tensor::arithmetic::add(
        &crate::tensor::arithmetic::add(
            &crate::tensor::arithmetic::add(&identity_like(n), &crate::tensor::arithmetic::mul_scalar(&scaled, 0.5)),
            &crate::tensor::arithmetic::add(
                &crate::tensor::arithmetic::mul_scalar(&a_sq, 0.1388888888888889),
                &crate::tensor::arithmetic::mul_scalar(&crate::tensor::arithmetic::matmul(&scaled, &a_sq), 0.02314814814814815),
            ),
        ),
        &crate::tensor::arithmetic::add(
            &crate::tensor::arithmetic::mul_scalar(&a4, 0.00248015873015873),
            &crate::tensor::arithmetic::add(
                &crate::tensor::arithmetic::mul_scalar(&crate::tensor::arithmetic::matmul(&scaled, &a4), 0.0002755731922398589),
                &crate::tensor::arithmetic::mul_scalar(&a6, 2.08767569878681e-5),
            ),
        ),
    );

    // Result: result^s
    let mut result = u;
    for _ in 0..s {
        result = crate::tensor::arithmetic::matmul(&result, &result);
    }

    result
}

fn identity_like(n: usize) -> Tensor {
    Tensor::identity(n)
}

/// Matrix square root for positive definite matrices.
pub fn matrix_sqrt(a: &Tensor) -> Tensor {
    let eigh_result = eigh(a);
    let n = a.shape()[0];

    // Build D^(1/2) * V^T
    let mut sqrt_data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let sum: f64 = (0..n).map(|k| {
                let eigval = eigh_result.eigenvalues[k];
                if eigval > 0.0 {
                    eigval.sqrt() * eigh_result.eigenvectors.get_index(&[j, k])
                        * eigh_result.eigenvectors.get_index(&[i, k])
                } else { 0.0 }
            }).sum();
            sqrt_data[i * n + j] = sum;
        }
    }
    Tensor::new(sqrt_data, vec![n, n])
}

// =============================================================================
// Pseudoinverse (Moore-Penrose)
// =============================================================================

/// Computes the Moore-Penrose pseudoinverse.
pub fn pinv(a: &Tensor) -> Tensor {
    let svd = svd_symmetric(a);
    let n = a.shape()[0];
    let tolerance = n as f64 * svd.singular_values.first().copied().unwrap_or(0.0).max(1e-10) * 1e-10;

    // Compute A^+ = V * D^+ * U^T
    let mut pinv_data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let sum: f64 = (0..n).filter(|&k| svd.singular_values[k] > tolerance)
                .map(|k| {
                    let s_inv = 1.0 / svd.singular_values[k];
                    s_inv * svd.v.get_index(&[i, k]) * svd.u.get_index(&[j, k])
                }).sum();
            pinv_data[i * n + j] = sum;
        }
    }
    Tensor::new(pinv_data, vec![n, n])
}

// =============================================================================
// Matrix Rank and Condition Number
// =============================================================================

/// Computes the matrix rank using SVD.
pub fn rank(a: &Tensor) -> usize {
    let svd = svd_symmetric(a);
    let n = a.shape()[0];
    let tol = n as f64 * svd.singular_values.first().copied().unwrap_or(0.0).max(1e-10) * 1e-10;
    svd.singular_values.iter().filter(|&&s| s > tol).count()
}

/// Computes the condition number (ratio of largest to smallest singular value).
pub fn cond(a: &Tensor) -> f64 {
    let svd = svd_symmetric(a);
    let max_s = svd.singular_values.first().copied().unwrap_or(0.0);
    let min_s = svd.singular_values.last().copied().unwrap_or(0.0);
    if min_s.abs() < 1e-12 { f64::INFINITY } else { max_s / min_s }
}

// =============================================================================
// Additional Matrix Utilities
// =============================================================================

/// Computes the outer product of two column vectors: a * b^T.
pub fn outer_product(a: &Tensor, b: &Tensor) -> Tensor {
    crate::tensor::arithmetic::outer(a, b)
}

/// Extracts the lower triangular portion of a matrix.
pub fn tril_fn(a: &Tensor, k: isize) -> Tensor {
    a.tril(k)
}

/// Extracts the upper triangular portion of a matrix.
pub fn triu_fn(a: &Tensor, k: isize) -> Tensor {
    a.triu(k)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_l1() {
        let a = Tensor::from_slice(&[-1.0, 2.0, -3.0], vec![3]);
        assert_eq!(norm_l1(&a), 6.0);
    }

    #[test]
    fn test_norm_l2() {
        let a = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        assert!((norm_l2(&a) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_linf() {
        let a = Tensor::from_slice(&[-1.0, 5.0, -3.0], vec![3]);
        assert_eq!(norm_linf(&a), 5.0);
    }

    #[test]
    fn test_norm_p() {
        let a = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        assert!((norm_p(&a, 3.0) - (27.0 + 64.0_f64).powf(1.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_trace() {
        let a = Tensor::identity(3);
        assert_eq!(trace(&a), 3.0);
    }

    #[test]
    fn test_diag() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        let d = diag(&a);
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d.get(0), 1.0);
        assert_eq!(d.get(1), 5.0);
        assert_eq!(d.get(2), 9.0);
    }

    #[test]
    fn test_det_1x1() {
        let a = Tensor::from_slice(&[5.0], vec![1, 1]);
        assert_eq!(det(&a), 5.0);
    }

    #[test]
    fn test_det_2x2() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(det(&a), -2.0);
    }

    #[test]
    fn test_det_3x3() {
        let a = Tensor::from_slice(&[6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0], vec![3, 3]);
        assert_eq!(det(&a), -306.0);
    }

    #[test]
    fn test_det_identity() {
        for n in [1, 2, 3, 5, 10] {
            assert!((det(&Tensor::identity(n)) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_inv_identity() {
        let a = Tensor::identity(3);
        let ainv = inv(&a);
        for i in 0..9 { assert!((ainv.get(i) - a.get(i)).abs() < 1e-10); }
    }

    #[test]
    fn test_inv_product() {
        let a = Tensor::from_slice(&[4.0, 7.0, 2.0, 6.0], vec![2, 2]);
        let ainv = inv(&a);
        let prod = crate::tensor::arithmetic::matmul(&a, &ainv);
        let eye = Tensor::identity(2);
        for i in 0..4 { assert!((prod.get(i) - eye.get(i)).abs() < 1e-10); }
    }

    #[test]
    fn test_inv_3x3() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0], vec![3, 3]);
        let ainv = inv(&a);
        let prod = crate::tensor::arithmetic::matmul(&a, &ainv);
        let eye = Tensor::identity(3);
        for i in 0..9 { assert!((prod.get(i) - eye.get(i)).abs() < 1e-10, "A*A^-1[{}] = {} != {}", i, prod.get(i), eye.get(i)); }
    }

    #[test]
    fn test_solve_identity() {
        let a = Tensor::identity(3);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let x = solve(&a, &b);
        for i in 0..3 { assert!((x.get(i) - b.get(i)).abs() < 1e-10); }
    }

    #[test]
    fn test_solve_2x2() {
        let a = Tensor::from_slice(&[2.0, 1.0, 5.0, 3.0], vec![2, 2]);
        let b = Tensor::from_slice(&[11.0, 27.0], vec![2]);
        let x = solve(&a, &b);
        // 2x + y = 11 => x = 6, y = -1
        // 5x + 3y = 27
        assert!((x.get(0) - 6.0).abs() < 1e-10);
        assert!((x.get(1) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_lu_decompose() {
        let a = Tensor::from_slice(&[2.0, 1.0, 6.0, 1.0, 3.0, 1.0, 1.0, 0.0, 5.0], vec![3, 3]);
        let lu = lu_decompose(&a);
        // Verify LU = PA
        let product = crate::tensor::arithmetic::matmul(&lu.l, &lu.u);
        for i in 0..9 {
            let pi = lu.piv.iter().position(|&p| p == i / 3).unwrap();
            assert!((product.get(i) - a.get_index(&[pi, i % 3])).abs() < 1e-10);
        }
    }

    #[test]
    fn test_qr_decompose() {
        let a = Tensor::from_slice(&[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0], vec![3, 3]);
        let qr = qr_decompose(&a);
        let product = crate::tensor::arithmetic::matmul(&qr.q, &qr.r);
        for i in 0..9 { assert!((product.get(i) - a.get(i)).abs() < 1e-8, "QR[{}] = {} vs {}", i, product.get(i), a.get(i)); }
    }

    #[test]
    fn test_cholesky() {
        let a = Tensor::from_slice(&[4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0], vec![3, 3]);
        let l = cholesky(&a);
        let lt = l.transpose();
        let product = crate::tensor::arithmetic::matmul(&l, &lt);
        for i in 0..9 { assert!((product.get(i) - a.get(i)).abs() < 1e-8); }
    }

    #[test]
    fn test_eigh_symmetric() {
        let a = Tensor::from_slice(&[2.0, 1.0, 1.0, 3.0], vec![2, 2]);
        let result = eigh(&a);
        assert_eq!(result.eigenvalues.len(), 2);
        // Verify A*v = lambda*v
        for (i, &lambda) in result.eigenvalues.iter().enumerate() {
            let v = vec![result.eigenvectors.get_index(&[0, i]), result.eigenvectors.get_index(&[1, i])];
            let av = vec![
                a.get_index(&[0, 0]) * v[0] + a.get_index(&[0, 1]) * v[1],
                a.get_index(&[1, 0]) * v[0] + a.get_index(&[1, 1]) * v[1],
            ];
            assert!((av[0] - lambda * v[0]).abs() < 1e-8);
            assert!((av[1] - lambda * v[1]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_svd_symmetric() {
        let a = Tensor::from_slice(&[1.0, 0.5, 0.5, 1.0], vec![2, 2]);
        let svd = svd_symmetric(&a);
        assert_eq!(svd.singular_values.len(), 2);
        assert!(svd.singular_values[0] >= svd.singular_values[1]);
    }

    #[test]
    fn test_matrix_power() {
        let a = Tensor::from_slice(&[1.0, 1.0, 0.0, 1.0], vec![2, 2]);
        let a3 = matrix_power(&a, 3);
        // (1 1)^3 = (1 3)
        // (0 1)    (0 1)
        assert!((a3.get_index(&[0, 1]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_power_zero() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0, 5.0], vec![2, 2]);
        let a0 = matrix_power(&a, 0);
        let eye = Tensor::identity(2);
        for i in 0..4 { assert!((a0.get(i) - eye.get(i)).abs() < 1e-10); }
    }

    #[test]
    fn test_matrix_power_negative() {
        let a = Tensor::from_slice(&[2.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let a_neg1 = matrix_power(&a, -1);
        let ainv = inv(&a);
        for i in 0..4 { assert!((a_neg1.get(i) - ainv.get(i)).abs() < 1e-10); }
    }

    #[test]
    fn test_pinv() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        let ap = pinv(&a);
        // A * A+ * A should be approximately A
        let aap = crate::tensor::arithmetic::matmul(&a, &ap);
        let aapa = crate::tensor::arithmetic::matmul(&aap, &a);
        for i in 0..9 { assert!((aapa.get(i) - a.get(i)).abs() < 1e-6, "pinv[{}] = {} vs {}", i, aapa.get(i), a.get(i)); }
    }

    #[test]
    fn test_rank() {
        let a = Tensor::identity(3);
        assert_eq!(rank(&a), 3);
    }

    #[test]
    fn test_cond_identity() {
        let a = Tensor::identity(3);
        assert!((cond(&a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_det_product() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let ab = crate::tensor::arithmetic::matmul(&a, &b);
        assert!((det(&ab) - det(&a) * det(&b)).abs() < 1e-10);
    }

    #[test]
    fn test_norm_frobenius() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!((norm_frobenius(&a) - expected).abs() < 1e-10);
    }
}
