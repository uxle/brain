//! # Numerically Stable Loss Operations
//!
//! Log-sum-exp, fused log-softmax, softmax, NLL, and one-hot encoding helpers.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Numerically stable log-sum-exp along the last dimension of a 2D tensor.
pub fn log_sum_exp_2d(logits: &Tensor) -> Vec<f64> {
    let shape = logits.shape();
    let rows = shape[0];
    let cols = if shape.len() > 1 { shape[1] } else { 1 };
    let data = logits.to_vec();

    let mut lse = vec![0.0f64; rows];
    for r in 0..rows {
        let row_slice = &data[r * cols..(r + 1) * cols];
        let max_val = row_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row_slice.iter().map(|&x| (x - max_val).exp()).sum();
        lse[r] = max_val + sum_exp.ln();
    }
    lse
}

/// Fused log-softmax along the last dimension.
pub fn log_softmax(logits: &Tensor) -> Tensor {
    let shape = logits.shape();
    let rows = shape[0];
    let cols = if shape.len() > 1 { shape[1] } else { 1 };
    let data = logits.to_vec();
    let lse = log_sum_exp_2d(logits);

    let mut out = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = data[r * cols + c] - lse[r];
        }
    }
    Tensor::from_vec(out, shape.to_vec())
}

/// Numerically stable softmax along the last dimension.
pub fn softmax(logits: &Tensor) -> Tensor {
    let log_s = log_softmax(logits);
    let data: Vec<f64> = log_s.to_vec().iter().map(|&v| v.exp()).collect();
    Tensor::from_vec(data, logits.shape().to_vec())
}

/// Negative Log Likelihood (NLL) loss given log-probabilities and target class indices.
pub fn nll_loss(log_probs: &Tensor, targets: &[usize]) -> Vec<f64> {
    let shape = log_probs.shape();
    let rows = shape[0];
    let cols = if shape.len() > 1 { shape[1] } else { 1 };
    let data = log_probs.to_vec();

    let n = rows.min(targets.len());
    let mut losses = vec![0.0f64; n];
    for r in 0..n {
        let c = targets[r];
        if c < cols {
            losses[r] = -data[r * cols + c];
        }
    }
    losses
}

/// One-hot encodes target indices into a 2D float tensor with optional label smoothing.
pub fn one_hot_target(targets: &[usize], num_classes: usize, smoothing: f64) -> Tensor {
    let n = targets.len();
    let mut one_hot = vec![smoothing / num_classes as f64; n * num_classes];
    let main_weight = 1.0 - smoothing + (smoothing / num_classes as f64);

    for (r, &c) in targets.iter().enumerate() {
        if c < num_classes {
            one_hot[r * num_classes + c] = main_weight;
        }
    }
    Tensor::from_vec(one_hot, vec![n, num_classes])
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
