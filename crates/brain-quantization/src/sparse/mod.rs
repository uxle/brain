//! # Sparse Matrix Representation
//!
//! Compressed Sparse Row (CSR), Compressed Sparse Column (CSC), and Coordinate (COO) sparse matrix representations.
#![allow(
    missing_docs,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

pub mod ops;

use super::core::{QuantError, QuantResult};
use brain_core::Tensor;

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
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
