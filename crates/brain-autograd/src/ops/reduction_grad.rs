//! # Reduction Operation Gradients
//!
//! Backward rules for axis-wise reductions: `sum`, `mean`, `var`, `std`, `norm`, `cumsum`, and `cumprod`.

use brain_core::tensor::broadcast as bcast_t;
use brain_core::{BrainResult, Tensor};

/// Backward for `sum_axis`: expands gradient along reduced axis.
pub fn grad_sum_axis(g: &Tensor, orig_shape: &[usize], axis: usize) -> BrainResult<Tensor> {
    let mut expanded = g.clone();
    if g.ndim() < orig_shape.len() {
        expanded = g.unsqueeze(axis);
    }
    bcast_t::broadcast_to(&expanded, orig_shape)
}

/// Backward for `mean_axis`: expands gradient and divides by reduction dimension size.
pub fn grad_mean_axis(g: &Tensor, orig_shape: &[usize], axis: usize) -> BrainResult<Tensor> {
    let unscaled = grad_sum_axis(g, orig_shape, axis)?;
    let dim_size = orig_shape[axis] as f64;
    Ok(unscaled.map(|x| x / dim_size))
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
