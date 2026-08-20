//! # Federated Compute Utilities
//!
//! Matrix operations, gradient computation helpers, and batched tensor ops.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Performs element-wise multiply-accumulate across a list of tensors.
pub fn multiply_accumulate(tensors: &[Tensor], scale: f64) -> Tensor {
    if tensors.is_empty() {
        return Tensor::scalar(0.0);
    }
    let s = Tensor::scalar(scale);
    tensors
        .iter()
        .fold(Tensor::zeros(tensors[0].shape().to_vec()), |acc, t| {
            &acc + &(t * &s)
        })
}

/// Computes the global gradient norm across all tensors.
pub fn global_grad_norm(tensors: &[Tensor]) -> f64 {
    tensors
        .iter()
        .flat_map(|t| t.to_vec())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

/// Clips gradients globally by their L2 norm.
pub fn clip_grad_norm(tensors: &mut [Tensor], max_norm: f64) {
    let norm = global_grad_norm(tensors);
    if norm > max_norm {
        let scale = Tensor::scalar(max_norm / norm);
        for t in tensors.iter_mut() {
            *t = &*t * &scale;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
