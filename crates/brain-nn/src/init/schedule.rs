//! # Deep Residual Initialization Schedules
//!
//! GPT-2/3 style scaled residual projections: 1/sqrt(2 * num_layers) and zero-initialization for residual gates.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Policy describing layer-specific initialization scaling.
#[derive(Debug, Clone, Copy, Default)]
pub struct InitPolicy {
    pub num_residual_layers: usize,
}

/// Scales residual branch output projection weights by 1 / sqrt(2 * num_residual_layers).
pub fn scaled_residual_init(weight: &Tensor, num_residual_layers: usize) -> Tensor {
    let scale = 1.0 / (2.0 * num_residual_layers.max(1) as f64).sqrt();
    weight * &Tensor::scalar(scale)
}

/// Initializes the last layer/projection of a residual block to exact zeros (identity pass-through at step 0).
pub fn zero_init_last_layer(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape.to_vec())
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
