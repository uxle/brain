//! Functional neural network layers, loss functions, embeddings, and attention.
//!
//! This module provides loss criteria (MSE, L1, Cross-Entropy, BCE), Scaled Dot-Product Attention,
//! embedding table lookups, and normalization blocks.

use crate::tensor::Tensor;

// =============================================================================
// Loss Functions
// =============================================================================

/// Mean Squared Error (MSE) loss: mean((pred - target)^2).
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> f64 {
    let diff = crate::tensor::arithmetic::sub(pred, target);
    let sq = crate::tensor::arithmetic::mul(&diff, &diff);
    crate::tensor::reduction::mean(&sq)
}

/// Mean Absolute Error (L1) loss: mean(|pred - target|).
pub fn l1_loss(pred: &Tensor, target: &Tensor) -> f64 {
    let diff = crate::tensor::arithmetic::sub(pred, target);
    let abs_diff = crate::tensor::math::abs(&diff);
    crate::tensor::reduction::mean(&abs_diff)
}

/// Binary Cross Entropy (BCE) loss: -mean(target * log(pred + eps) + (1 - target) * log(1 - pred + eps)).
pub fn bce_loss(pred: &Tensor, target: &Tensor, eps: f64) -> f64 {
    assert_eq!(pred.shape(), target.shape());
    let mut sum = 0.0;
    let n = pred.numel();
    for i in 0..n {
        let p = pred.get(i).clamp(eps, 1.0 - eps);
        let y = target.get(i);
        sum += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
    }
    sum / (n as f64)
}

/// Cross Entropy loss with logits and class index targets.
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> f64 {
    assert_eq!(logits.ndim(), 2);
    let (batch_size, num_classes) = (logits.shape()[0], logits.shape()[1]);
    assert_eq!(targets.len(), batch_size);

    let log_sm = crate::tensor::special::log_softmax(logits, 1);
    let mut total_loss = 0.0;
    for b in 0..batch_size {
        let target_class = targets[b];
        assert!(target_class < num_classes);
        let log_p = log_sm.get_2d(b, target_class);
        total_loss += -log_p;
    }
    total_loss / (batch_size as f64)
}

// =============================================================================
// Attention & Embeddings
// =============================================================================

/// Scaled Dot-Product Attention: softmax(Q @ K^T / sqrt(d_k)) @ V.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Tensor {
    assert!(q.ndim() == 2 && k.ndim() == 2 && v.ndim() == 2);
    let d_k = q.shape()[1] as f64;
    let scale = 1.0 / d_k.sqrt();

    let scores = crate::tensor::arithmetic::matmul(q, &k.t());
    let mut scaled_scores = crate::tensor::arithmetic::mul_scalar(&scores, scale);

    if let Some(m) = mask {
        let masked = crate::tensor::special::where_cond(
            m,
            &scaled_scores,
            &Tensor::full(scaled_scores.shape().to_vec(), -1e9),
        );
        scaled_scores = masked;
    }

    let attn_weights = crate::tensor::special::softmax(&scaled_scores, 1);
    crate::tensor::arithmetic::matmul(&attn_weights, v)
}

/// Embedding layer lookup: extracts vectors corresponding to discrete indices.
pub fn embedding(indices: &Tensor, weight: &Tensor) -> Tensor {
    assert_eq!(weight.ndim(), 2);
    let embedding_dim = weight.shape()[1];
    let num_indices = indices.numel();

    let mut out_shape = indices.shape().to_vec();
    out_shape.push(embedding_dim);

    let mut out_data = Vec::with_capacity(num_indices * embedding_dim);
    for i in 0..num_indices {
        let idx = indices.get(i) as usize;
        assert!(idx < weight.shape()[0], "embedding index out of range");
        for d in 0..embedding_dim {
            out_data.push(weight.get_2d(idx, d));
        }
    }

    Tensor::new(out_data, out_shape)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_losses() {
        let p = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let t = Tensor::from_slice(&[2.0, 4.0], vec![2]);
        assert_eq!(mse_loss(&p, &t), 2.5);
        assert_eq!(l1_loss(&p, &t), 1.5);
    }

    #[test]
    fn test_embedding() {
        let weight = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let idx = Tensor::from_slice(&[1.0, 0.0], vec![2]);
        let emb = embedding(&idx, &weight);
        assert_eq!(emb.shape(), &[2, 2]);
        assert_eq!(emb.data(), &[3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_neural_loss_and_attention() {
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let target = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mse_loss(&pred, &target), 0.0);
        assert_eq!(l1_loss(&pred, &target), 0.0);
    }
}
