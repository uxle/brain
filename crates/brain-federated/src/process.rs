//! # Federated Processing Pipeline
//!
//! Post-processing of aggregated weights, scheduling, and evaluation helpers.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Applies a learning rate schedule (cosine decay) to a given step.
pub fn cosine_lr(base_lr: f64, step: usize, total_steps: usize) -> f64 {
    let t = step as f64 / total_steps.max(1) as f64;
    base_lr * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
}

/// Applies L2 weight decay to a tensor.
pub fn apply_weight_decay(t: &Tensor, weight_decay: f64) -> Tensor {
    let wd = Tensor::scalar(1.0 - weight_decay);
    t * &wd
}

/// Evaluates a simple mean-squared error given predictions and targets.
pub fn mse_eval(predictions: &[f64], targets: &[f64]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / predictions.len() as f64
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
