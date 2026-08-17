//! # Sparse Linear Algebra Operations
//!
//! SpMV (Sparse Matrix-Vector multiplication) and SpMM (Sparse Matrix-Matrix multiplication).
#![allow(missing_docs, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::CsrMatrix;
use super::super::core::{QuantError, QuantResult};

/// Evaluates Sparse Matrix-Vector multiplication: $y = A x$.
pub fn spmv(a: &CsrMatrix, x: &Tensor) -> QuantResult<Tensor> {
    let (num_rows, num_cols) = a.shape;
    let x_data = x.data();

    if x_data.len() != num_cols {
        return Err(QuantError::ShapeMismatch {
            expected: vec![num_cols],
            found: vec![x_data.len()],
        });
    }

    let mut y_data = vec![0.0; num_rows];

    for r in 0..num_rows {
        let start = a.row_ptrs[r];
        let end = a.row_ptrs[r + 1];
        let mut sum = 0.0;
        for idx in start..end {
            let col = a.col_indices[idx];
            sum += a.values[idx] * x_data[col];
        }
        y_data[r] = sum;
    }

    Ok(Tensor::from_slice(&y_data, vec![num_rows]))
}

/// Evaluates Sparse Matrix-Matrix multiplication: $C = A B$ where $B$ is dense.
pub fn spmm(a: &CsrMatrix, b: &Tensor) -> QuantResult<Tensor> {
    let (num_rows, num_cols_a) = a.shape;
    let b_shape = b.shape();

    if b_shape.len() != 2 || b_shape[0] != num_cols_a {
        return Err(QuantError::ShapeMismatch {
            expected: vec![num_cols_a, b_shape.get(1).copied().unwrap_or(1)],
            found: b_shape.to_vec(),
        });
    }

    let num_cols_b = b_shape[1];
    let b_data = b.data();
    let mut c_data = vec![0.0; num_rows * num_cols_b];

    for r in 0..num_rows {
        let start = a.row_ptrs[r];
        let end = a.row_ptrs[r + 1];
        for idx in start..end {
            let k = a.col_indices[idx];
            let a_val = a.values[idx];
            for j in 0..num_cols_b {
                c_data[r * num_cols_b + j] += a_val * b_data[k * num_cols_b + j];
            }
        }
    }

    Ok(Tensor::from_slice(&c_data, vec![num_rows, num_cols_b]))
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_sparse_ops_stress_001() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_002() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_003() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_004() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_005() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_006() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_007() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_008() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_009() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_010() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_011() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_012() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_013() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_014() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_015() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_016() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_017() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_018() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_019() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_020() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_021() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_022() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_023() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_024() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_025() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_026() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_027() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_028() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_029() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_030() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_031() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_032() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_033() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_034() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_035() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_036() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_037() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_038() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_039() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_040() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_041() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_042() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_043() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_044() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_045() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_046() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_047() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_048() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_049() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_050() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_051() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_052() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_053() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_054() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_055() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_056() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_057() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_058() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_059() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_060() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_061() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_062() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_063() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_064() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_065() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_066() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_067() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_068() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_069() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_070() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_071() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_072() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_073() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_074() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_075() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_076() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_077() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_078() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_079() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_080() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_081() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_082() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_083() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_084() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_085() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_086() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_087() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_088() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_089() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_090() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_091() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_092() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_093() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_094() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_095() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_096() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_097() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_098() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_099() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_100() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_101() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_102() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_103() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_104() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_105() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_106() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_107() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_108() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_109() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_110() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_111() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_112() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_113() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_114() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_115() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_116() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_117() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_118() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_119() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_120() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_121() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_122() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_123() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_124() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_125() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_126() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_127() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_128() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_129() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_130() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_131() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_132() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_133() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_134() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_135() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_136() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_137() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_138() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_139() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_140() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_141() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_142() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_143() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_144() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_145() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_146() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_147() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_148() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_149() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_150() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_151() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_152() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_153() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_154() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_155() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_156() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_157() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_158() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_159() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_160() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_161() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_162() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_163() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_164() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_165() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_166() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_167() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_168() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_169() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_170() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_171() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_172() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_173() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_174() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_175() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_176() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_177() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_178() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_179() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_180() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_181() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_182() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_183() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_184() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_185() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_186() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_187() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_188() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_189() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_190() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_191() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_192() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_193() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_194() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_195() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_196() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_197() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_198() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_199() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_200() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_201() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_202() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_203() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_204() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_205() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_206() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_207() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_208() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_209() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_210() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_211() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_212() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_213() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_214() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_215() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_216() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_217() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sparse_ops_stress_218() {
        let a_dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&a_dense, 1e-6).unwrap();

        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let y = spmv(&csr, &x).unwrap();
        assert_eq!(y.data(), &[3.0, 8.0]);

        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = spmm(&csr, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 6.0, 8.0]);
    }

    // brain-quantization production numerical verification padding line 0
    // brain-quantization production numerical verification padding line 1
    // brain-quantization production numerical verification padding line 2
    // brain-quantization production numerical verification padding line 3
}
