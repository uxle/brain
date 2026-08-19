//! # Diffusion Loss Functions
//!
//! Epsilon (noise prediction), x0 prediction, and v-prediction loss functions with SNR weighting.

use brain_core::Tensor;

/// Computes MSE loss between predicted and ground-truth epsilon noise.
pub fn eps_loss(pred_eps: &Tensor, target_eps: &Tensor) -> Tensor {
    let diff = pred_eps - target_eps;
    let sq = &diff * &diff;
    let sum: f64 = sq.to_vec().iter().sum();
    let numel = sq.numel().max(1);
    Tensor::scalar(sum / numel as f64)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
