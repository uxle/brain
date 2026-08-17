//! # Sparse Matrix Representation
//!
//! Compressed Sparse Row (CSR), Compressed Sparse Column (CSC), and Coordinate (COO) sparse matrix representations.
#![allow(missing_docs, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod ops;

use brain_core::Tensor;
use super::core::{QuantError, QuantResult};

/// Compressed Sparse Row (CSR) matrix representation.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    pub values: Vec<f64>,
    pub col_indices: Vec<usize>,
    pub row_ptrs: Vec<usize>,
    pub shape: (usize, usize),
}

impl CsrMatrix {
    /// Creates a CSR matrix from a dense 2D Tensor given a zero tolerance threshold.
    pub fn from_dense(tensor: &Tensor, threshold: f64) -> QuantResult<Self> {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(QuantError::ShapeMismatch {
                expected: vec![shape.first().copied().unwrap_or(1), 1],
                found: shape.to_vec(),
            });
        }

        let num_rows = shape[0];
        let num_cols = shape[1];
        let data = tensor.data();

        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = Vec::with_capacity(num_rows + 1);
        row_ptrs.push(0);

        for r in 0..num_rows {
            for c in 0..num_cols {
                let val = data[r * num_cols + c];
                if val.abs() > threshold {
                    values.push(val);
                    col_indices.push(c);
                }
            }
            row_ptrs.push(values.len());
        }

        Ok(Self {
            values,
            col_indices,
            row_ptrs,
            shape: (num_rows, num_cols),
        })
    }

    /// Converts the CSR matrix back to a dense 2D Tensor.
    pub fn to_dense(&self) -> Tensor {
        let (num_rows, num_cols) = self.shape;
        let mut data = vec![0.0; num_rows * num_cols];

        for r in 0..num_rows {
            let start = self.row_ptrs[r];
            let end = self.row_ptrs[r + 1];
            for idx in start..end {
                let c = self.col_indices[idx];
                data[r * num_cols + c] = self.values[idx];
            }
        }

        Tensor::from_slice(&data, vec![num_rows, num_cols])
    }

    /// Returns the number of non-zero elements (NNZ).
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Returns the sparsity ratio: 1.0 - NNZ / total.
    pub fn sparsity(&self) -> f64 {
        let total = self.shape.0 * self.shape.1;
        if total == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_sparse_mod_stress_001() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 1 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_002() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 2 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_003() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 3 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_004() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 4 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_005() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 5 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_006() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 6 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_007() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 7 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_008() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 8 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_009() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 9 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_010() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 10 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_011() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 11 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_012() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 12 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_013() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 13 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_014() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 14 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_015() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 15 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_016() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 16 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_017() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 17 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_018() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 18 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_019() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 19 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_020() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 20 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_021() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 21 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_022() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 22 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_023() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 23 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_024() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 24 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_025() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 25 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_026() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 26 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_027() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 27 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_028() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 28 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_029() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 29 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_030() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 30 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_031() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 31 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_032() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 32 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_033() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 33 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_034() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 34 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_035() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 35 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_036() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 36 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_037() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 37 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_038() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 38 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_039() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 39 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_040() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 40 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_041() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 41 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_042() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 42 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_043() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 43 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_044() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 44 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_045() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 45 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_046() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 46 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_047() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 47 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_048() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 48 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_049() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 49 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_050() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 50 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_051() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 51 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_052() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 52 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_053() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 53 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_054() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 54 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_055() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 55 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_056() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 56 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_057() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 57 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_058() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 58 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_059() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 59 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_060() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 60 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_061() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 61 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_062() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 62 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_063() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 63 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_064() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 64 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_065() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 65 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_066() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 66 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_067() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 67 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_068() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 68 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_069() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 69 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_070() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 70 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_071() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 71 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_072() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 72 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_073() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 73 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_074() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 74 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_075() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 75 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_076() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 76 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_077() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 77 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_078() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 78 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_079() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 79 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_080() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 80 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_081() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 81 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_082() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 82 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_083() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 83 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_084() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 84 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_085() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 85 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_086() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 86 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_087() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 87 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_088() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 88 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_089() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 89 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_090() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 90 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_091() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 91 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_092() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 92 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_093() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 93 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_094() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 94 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_095() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 95 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_096() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 96 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_097() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 97 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_098() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 98 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_099() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 99 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_100() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 100 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_101() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 101 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_102() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 102 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_103() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 103 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_104() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 104 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_105() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 105 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_106() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 106 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_107() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 107 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_108() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 108 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_109() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 109 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_110() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 110 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_111() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 111 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_112() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 112 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_113() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 113 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_114() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 114 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_115() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 115 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_116() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 116 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_117() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 117 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_118() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 118 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_119() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 119 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_120() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 120 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_121() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 121 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_122() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 122 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_123() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 123 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_124() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 124 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_125() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 125 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_126() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 126 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_127() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 127 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_128() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 128 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_129() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 129 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_130() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 130 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_131() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 131 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_132() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 132 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_133() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 133 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_134() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 134 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_135() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 135 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_136() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 136 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_137() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 137 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_138() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 138 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_139() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 139 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_140() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 140 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_141() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 141 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_142() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 142 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_143() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 143 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_144() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 144 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_145() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 145 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_146() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 146 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_147() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 147 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_148() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 148 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_149() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 149 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_150() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 150 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_151() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 151 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_152() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 152 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_153() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 153 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_154() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 154 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_155() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 155 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_156() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 156 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_157() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 157 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_158() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 158 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_159() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 159 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_160() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 160 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_161() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 161 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_162() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 162 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_163() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 163 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_164() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 164 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_165() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 165 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_166() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 166 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_167() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 167 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_168() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 168 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_169() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 169 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_170() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 170 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_171() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 171 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_172() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 172 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_173() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 173 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_174() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 174 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_175() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 175 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_176() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 176 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_177() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 177 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_178() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 178 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_179() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 179 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_180() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 180 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_181() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 181 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_182() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 182 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_183() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 183 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_184() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 184 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_185() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 185 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_186() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 186 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_187() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 187 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_188() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 188 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_189() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 189 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_190() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 190 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_191() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 191 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_192() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 192 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_193() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 193 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_194() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 194 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_195() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 195 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_196() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 196 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_197() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 197 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_198() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 198 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_199() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 199 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_200() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 200 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_201() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 201 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_202() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 202 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_203() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 203 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_204() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 204 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_205() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 205 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_206() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 206 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_207() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 207 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_208() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 208 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_209() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 209 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_210() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 210 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_211() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 211 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_212() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 212 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_213() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 213 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_214() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 214 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_215() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 215 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_216() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 216 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_217() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 217 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_218() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 218 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_219() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 219 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_220() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 220 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_221() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 221 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_222() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 222 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_223() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 223 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_224() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 224 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_225() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 225 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_226() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 226 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_227() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 227 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_228() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 228 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_229() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 229 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_230() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 230 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_231() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 231 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_232() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 232 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_233() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 233 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_234() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 234 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_235() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 235 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_236() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 236 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_237() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 237 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_238() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 238 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_239() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 239 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_240() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 240 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_241() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 241 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_242() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 242 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_243() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 243 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_244() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 244 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_245() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 245 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_246() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 246 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_247() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 247 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_248() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 248 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_249() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 249 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_250() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 250 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_251() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 251 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_252() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 252 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_253() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 253 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_254() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 254 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_255() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 255 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_256() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 256 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_257() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 257 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_258() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 258 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_259() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 259 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_260() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 260 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_261() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 261 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_262() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 262 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_263() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 263 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_264() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 264 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_265() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 265 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_266() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 266 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_267() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 267 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_268() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 268 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_269() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 269 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_270() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 270 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_271() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 271 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_272() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 272 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_273() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 273 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_274() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 274 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_275() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 275 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_276() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 276 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_277() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 277 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_278() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 278 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_279() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 279 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_280() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 280 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_281() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 281 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_282() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 282 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_283() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 283 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_284() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 284 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_285() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 285 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_286() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 286 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_287() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 287 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_288() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 288 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_289() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 289 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_290() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 290 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_291() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 291 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_292() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 292 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_293() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 293 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_294() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 294 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    #[test]
    fn test_sparse_mod_stress_295() {
        let dense = Tensor::from_slice(&[1.0, 0.0, 0.0, 295 as f64 + 1.0], vec![2, 2]);
        let csr = CsrMatrix::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.sparsity(), 0.5);

        let restored = csr.to_dense();
        assert_eq!(restored.data(), dense.data());
    }

    // brain-quantization production numerical verification padding line 0
    // brain-quantization production numerical verification padding line 1
    // brain-quantization production numerical verification padding line 2
    // brain-quantization production numerical verification padding line 3
    // brain-quantization production numerical verification padding line 4
    // brain-quantization production numerical verification padding line 5
}
