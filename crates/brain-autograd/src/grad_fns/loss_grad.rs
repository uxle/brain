//! # Loss Function Gradients
//!
//! Differentiable rules for standard loss criteria:
//! MSE, Binary Cross-Entropy, Cross-Entropy, Huber loss.

use brain_core::tensor::arithmetic as arith_t;
use brain_core::tensor::math as math_t;
use brain_core::tensor::special as spec_t;
use brain_core::{BrainResult, Tensor};

/// Gradient of Mean Squared Error: `2 * (pred - target) / N * g`.
pub fn grad_mse_loss(pred: &Tensor, target: &Tensor, g: f64) -> BrainResult<Tensor> {
    let diff = arith_t::sub(pred, target);
    let n = pred.numel() as f64;
    let factor = 2.0 * g / n;
    Ok(diff.map(|x| x * factor))
}

/// Gradient of fused Cross-Entropy loss: `(softmax(logits) - target_one_hot) / N * g`.
pub fn grad_cross_entropy_logits(
    logits: &Tensor,
    target_indices: &[usize],
    g: f64,
) -> BrainResult<Tensor> {
    let sm = spec_t::softmax(logits, logits.ndim() - 1);
    let mut grad_data = sm.data().to_vec();
    let num_classes = logits.shape().last().copied().unwrap_or(1);
    let batch_size = logits.numel() / num_classes;

    for (b, &class_idx) in target_indices.iter().enumerate().take(batch_size) {
        if class_idx < num_classes {
            grad_data[b * num_classes + class_idx] -= 1.0;
        }
    }

    let factor = g / batch_size as f64;
    let out = Tensor::from_slice(&grad_data, logits.shape().to_vec()).map(|x| x * factor);
    Ok(out)
}

/// Gradient of Binary Cross-Entropy with Logits: `(sigmoid(logits) - targets) / N * g`.
pub fn grad_bce_with_logits(logits: &Tensor, targets: &Tensor, g: f64) -> BrainResult<Tensor> {
    let sig = math_t::sigmoid(logits);
    let diff = arith_t::sub(&sig, targets);
    let n = logits.numel() as f64;
    let factor = g / n;
    Ok(diff.map(|x| x * factor))
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
