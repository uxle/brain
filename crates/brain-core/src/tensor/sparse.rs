//! Compressed sparse matrix formats (COO, CSR) and sparse linear algebra kernels.
//!
//! This module provides [`SparseCOO`] and [`SparseCSR`] representations with SpMV and SpMM multiplication algorithms.

use crate::tensor::Tensor;

/// Sparse matrix in Coordinate (COO) format.
#[derive(Debug, Clone)]
pub struct SparseCOO {
    /// Row indices for non-zero entries.
    pub row_indices: Vec<usize>,
    /// Column indices for non-zero entries.
    pub col_indices: Vec<usize>,
    /// Values of non-zero entries.
    pub values: Vec<f64>,
    /// 2D shape (rows, cols).
    pub shape: (usize, usize),
}

impl SparseCOO {
    /// Creates a new COO sparse matrix.
    pub fn new(shape: (usize, usize)) -> Self {
        SparseCOO {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            shape,
        }
    }

    /// Inserts a non-zero element.
    pub fn insert(&mut self, row: usize, col: usize, val: f64) {
        assert!(row < self.shape.0 && col < self.shape.1);
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(val);
    }

    /// Converts the COO matrix to a dense [`Tensor`].
    pub fn to_dense(&self) -> Tensor {
        let mut out = Tensor::zeros(vec![self.shape.0, self.shape.1]);
        for i in 0..self.values.len() {
            let r = self.row_indices[i];
            let c = self.col_indices[i];
            let cur = out.get_2d(r, c);
            out.set_2d(r, c, cur + self.values[i]);
        }
        out
    }

    /// Sparse-dense matrix multiplication: out = self * dense_b.
    pub fn spmm(&self, b: &Tensor) -> Tensor {
        assert_eq!(b.ndim(), 2);
        assert_eq!(self.shape.1, b.shape()[0]);
        let n_cols = b.shape()[1];
        let mut out = Tensor::zeros(vec![self.shape.0, n_cols]);

        for idx in 0..self.values.len() {
            let r = self.row_indices[idx];
            let k = self.col_indices[idx];
            let v = self.values[idx];
            for j in 0..n_cols {
                let b_val = b.get_2d(k, j);
                let cur = out.get_2d(r, j);
                out.set_2d(r, j, cur + v * b_val);
            }
        }
        out
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_coo_basic() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 5.0);
        coo.insert(1, 1, 3.0);
        let dense = coo.to_dense();
        assert_eq!(dense.get_2d(0, 0), 5.0);
        assert_eq!(dense.get_2d(0, 1), 0.0);
        assert_eq!(dense.get_2d(1, 1), 3.0);
    }

    #[test]
    fn test_sparse_coo_basics() {
        let mut coo = SparseCOO::new((2, 3));
        coo.insert(0, 1, 5.0);
        coo.insert(1, 2, 10.0);
        let dense = coo.to_dense();
        assert_eq!(dense.shape(), &[2, 3]);
        assert_eq!(dense.get_2d(0, 1), 5.0);
        assert_eq!(dense.get_2d(1, 2), 10.0);
    }
}
