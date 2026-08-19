//! # Loss Utilities
//!
//! Reduction application, shape validation, numerical clamping, and weighted averages.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{Reduction, LossError, LossResult};

/// Applies reduction (Mean, Sum, None) to a vector of per-sample losses.
pub fn reduction_apply(losses: &[f64], reduction: Reduction) -> Tensor {
    if losses.is_empty() {
        return Tensor::zeros(vec![1]);
    }
    match reduction {
        Reduction::Mean => {
            let sum: f64 = losses.iter().sum();
            Tensor::from_vec(vec![sum / losses.len() as f64], vec![1])
        }
        Reduction::Sum => {
            let sum: f64 = losses.iter().sum();
            Tensor::from_vec(vec![sum], vec![1])
        }
        Reduction::None => {
            Tensor::from_vec(losses.to_vec(), vec![losses.len()])
        }
    }
}

/// Validates that two tensors have matching shapes.
pub fn check_shapes(a: &Tensor, b: &Tensor) -> LossResult<()> {
    if a.shape() != b.shape() {
        return Err(LossError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    Ok(())
}

/// Clamps values to avoid log(0) and division by zero: [eps, 1.0 - eps].
pub fn clamp_eps(val: f64, eps: f64) -> f64 {
    val.clamp(eps, 1.0 - eps)
}

/// Computes weighted average of sample losses.
pub fn weighted_average(losses: &[f64], weights: &[f64]) -> f64 {
    let n = losses.len().min(weights.len());
    if n == 0 { return 0.0; }
    let mut total_loss = 0.0;
    let mut total_weight = 0.0;
    for i in 0..n {
        total_loss += losses[i] * weights[i];
        total_weight += weights[i];
    }
    if total_weight > 0.0 { total_loss / total_weight } else { 0.0 }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
