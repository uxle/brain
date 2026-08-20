//! # Sparse Linear Algebra Operations
//!
//! SpMV (Sparse Matrix-Vector multiplication) and SpMM (Sparse Matrix-Matrix multiplication).
#![allow(
    missing_docs,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

use super::super::core::{QuantError, QuantResult};
use super::CsrMatrix;
use brain_core::Tensor;

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
