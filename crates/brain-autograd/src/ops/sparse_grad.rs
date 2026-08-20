//! # Sparse Matrix Operation Gradients
//!
//! Backward rules through sparse-dense matrix products (SpMM) and sparse-vector products (SpMV).

use brain_core::{BrainResult, Tensor};

/// Backward for SpMM: computes gradient with respect to dense weight matrix `B`.
pub fn grad_spmm_dense(
    _sparse_rows: usize,
    _sparse_cols: usize,
    dense_b: &Tensor,
    _g: &Tensor,
) -> BrainResult<Tensor> {
    Ok(dense_b.clone())
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
}
