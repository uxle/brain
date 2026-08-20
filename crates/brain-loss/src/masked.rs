//! # Masked Loss Wrappers
//!
//! Padding-aware and boolean-masked loss reductions for NLP & sequence processing.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Applies a boolean or float mask to sample losses before mean reduction.
pub fn apply_loss_mask(losses: &[f64], mask: &[bool]) -> Tensor {
    let n = losses.len().min(mask.len());
    if n == 0 {
        return Tensor::zeros(vec![1]);
    }

    let mut sum = 0.0f64;
    let mut count = 0usize;

    for i in 0..n {
        if mask[i] {
            sum += losses[i];
            count += 1;
        }
    }

    let avg = if count > 0 { sum / count as f64 } else { 0.0 };
    Tensor::from_vec(vec![avg], vec![1])
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
