//! # Broadcasting Gradient Computation
//!
//! Shape-aware gradient reduction rules for broadcast forward operations.

use brain_core::{BrainResult, Tensor};

/// Reduces incoming broadcast gradient `g` back to operand's original shape `target_shape`.
pub fn unbroadcast(g: &Tensor, target_shape: &[usize]) -> BrainResult<Tensor> {
    crate::grad_fns::util::sum_to_shape(g, target_shape)
}

/// Computes the backward pass for `broadcast_to(x, new_shape)`.
pub fn grad_of_broadcast_to(g: &Tensor, orig_shape: &[usize]) -> BrainResult<Tensor> {
    unbroadcast(g, orig_shape)
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
