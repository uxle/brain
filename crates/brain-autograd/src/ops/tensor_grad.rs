//! # View & Structural Tensor Operation Gradients
//!
//! Backward rules for shape and view transformations:
//! `transpose`, `permute`, `reshape`, `narrow`, `flip`, `pad`, `unfold`.

use brain_core::{BrainResult, Tensor};

/// Backward for `reshape`: restores original shape.
pub fn grad_reshape(g: &Tensor, orig_shape: &[usize]) -> BrainResult<Tensor> {
    Ok(g.reshape(orig_shape.to_vec()))
}

/// Backward for `transpose`: applies transpose on identical dimensions.
pub fn grad_transpose(g: &Tensor, dim0: usize, dim1: usize) -> BrainResult<Tensor> {
    Ok(g.transpose(dim0, dim1))
}

/// Backward for `permute`: applies inverse permutation.
pub fn grad_permute(g: &Tensor, dims: &[usize]) -> BrainResult<Tensor> {
    let mut inv = vec![0; dims.len()];
    for (i, &d) in dims.iter().enumerate() {
        inv[d] = i;
    }
    Ok(g.permute(&inv))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;
}
