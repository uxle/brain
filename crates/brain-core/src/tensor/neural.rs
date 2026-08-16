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
    fn test_neural_stress_case_001() {
        let p = Tensor::from_slice(&[1.0], vec![1]);
        let t = Tensor::from_slice(&[2.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_002() {
        let p = Tensor::from_slice(&[2.0], vec![1]);
        let t = Tensor::from_slice(&[3.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_003() {
        let p = Tensor::from_slice(&[3.0], vec![1]);
        let t = Tensor::from_slice(&[4.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_004() {
        let p = Tensor::from_slice(&[4.0], vec![1]);
        let t = Tensor::from_slice(&[5.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_005() {
        let p = Tensor::from_slice(&[5.0], vec![1]);
        let t = Tensor::from_slice(&[6.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_006() {
        let p = Tensor::from_slice(&[6.0], vec![1]);
        let t = Tensor::from_slice(&[7.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_007() {
        let p = Tensor::from_slice(&[7.0], vec![1]);
        let t = Tensor::from_slice(&[8.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_008() {
        let p = Tensor::from_slice(&[8.0], vec![1]);
        let t = Tensor::from_slice(&[9.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_009() {
        let p = Tensor::from_slice(&[9.0], vec![1]);
        let t = Tensor::from_slice(&[10.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_010() {
        let p = Tensor::from_slice(&[10.0], vec![1]);
        let t = Tensor::from_slice(&[11.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_011() {
        let p = Tensor::from_slice(&[11.0], vec![1]);
        let t = Tensor::from_slice(&[12.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_012() {
        let p = Tensor::from_slice(&[12.0], vec![1]);
        let t = Tensor::from_slice(&[13.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_013() {
        let p = Tensor::from_slice(&[13.0], vec![1]);
        let t = Tensor::from_slice(&[14.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_014() {
        let p = Tensor::from_slice(&[14.0], vec![1]);
        let t = Tensor::from_slice(&[15.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_015() {
        let p = Tensor::from_slice(&[15.0], vec![1]);
        let t = Tensor::from_slice(&[16.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_016() {
        let p = Tensor::from_slice(&[16.0], vec![1]);
        let t = Tensor::from_slice(&[17.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_017() {
        let p = Tensor::from_slice(&[17.0], vec![1]);
        let t = Tensor::from_slice(&[18.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_018() {
        let p = Tensor::from_slice(&[18.0], vec![1]);
        let t = Tensor::from_slice(&[19.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_019() {
        let p = Tensor::from_slice(&[19.0], vec![1]);
        let t = Tensor::from_slice(&[20.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_020() {
        let p = Tensor::from_slice(&[20.0], vec![1]);
        let t = Tensor::from_slice(&[21.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_021() {
        let p = Tensor::from_slice(&[21.0], vec![1]);
        let t = Tensor::from_slice(&[22.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_022() {
        let p = Tensor::from_slice(&[22.0], vec![1]);
        let t = Tensor::from_slice(&[23.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_023() {
        let p = Tensor::from_slice(&[23.0], vec![1]);
        let t = Tensor::from_slice(&[24.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_024() {
        let p = Tensor::from_slice(&[24.0], vec![1]);
        let t = Tensor::from_slice(&[25.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_025() {
        let p = Tensor::from_slice(&[25.0], vec![1]);
        let t = Tensor::from_slice(&[26.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_026() {
        let p = Tensor::from_slice(&[26.0], vec![1]);
        let t = Tensor::from_slice(&[27.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_027() {
        let p = Tensor::from_slice(&[27.0], vec![1]);
        let t = Tensor::from_slice(&[28.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_028() {
        let p = Tensor::from_slice(&[28.0], vec![1]);
        let t = Tensor::from_slice(&[29.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_029() {
        let p = Tensor::from_slice(&[29.0], vec![1]);
        let t = Tensor::from_slice(&[30.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_030() {
        let p = Tensor::from_slice(&[30.0], vec![1]);
        let t = Tensor::from_slice(&[31.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_031() {
        let p = Tensor::from_slice(&[31.0], vec![1]);
        let t = Tensor::from_slice(&[32.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_032() {
        let p = Tensor::from_slice(&[32.0], vec![1]);
        let t = Tensor::from_slice(&[33.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_033() {
        let p = Tensor::from_slice(&[33.0], vec![1]);
        let t = Tensor::from_slice(&[34.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_034() {
        let p = Tensor::from_slice(&[34.0], vec![1]);
        let t = Tensor::from_slice(&[35.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_035() {
        let p = Tensor::from_slice(&[35.0], vec![1]);
        let t = Tensor::from_slice(&[36.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_036() {
        let p = Tensor::from_slice(&[36.0], vec![1]);
        let t = Tensor::from_slice(&[37.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_037() {
        let p = Tensor::from_slice(&[37.0], vec![1]);
        let t = Tensor::from_slice(&[38.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_038() {
        let p = Tensor::from_slice(&[38.0], vec![1]);
        let t = Tensor::from_slice(&[39.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_039() {
        let p = Tensor::from_slice(&[39.0], vec![1]);
        let t = Tensor::from_slice(&[40.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_040() {
        let p = Tensor::from_slice(&[40.0], vec![1]);
        let t = Tensor::from_slice(&[41.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_041() {
        let p = Tensor::from_slice(&[41.0], vec![1]);
        let t = Tensor::from_slice(&[42.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_042() {
        let p = Tensor::from_slice(&[42.0], vec![1]);
        let t = Tensor::from_slice(&[43.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_043() {
        let p = Tensor::from_slice(&[43.0], vec![1]);
        let t = Tensor::from_slice(&[44.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_044() {
        let p = Tensor::from_slice(&[44.0], vec![1]);
        let t = Tensor::from_slice(&[45.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_045() {
        let p = Tensor::from_slice(&[45.0], vec![1]);
        let t = Tensor::from_slice(&[46.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_046() {
        let p = Tensor::from_slice(&[46.0], vec![1]);
        let t = Tensor::from_slice(&[47.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_047() {
        let p = Tensor::from_slice(&[47.0], vec![1]);
        let t = Tensor::from_slice(&[48.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_048() {
        let p = Tensor::from_slice(&[48.0], vec![1]);
        let t = Tensor::from_slice(&[49.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_049() {
        let p = Tensor::from_slice(&[49.0], vec![1]);
        let t = Tensor::from_slice(&[50.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_050() {
        let p = Tensor::from_slice(&[50.0], vec![1]);
        let t = Tensor::from_slice(&[51.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_051() {
        let p = Tensor::from_slice(&[51.0], vec![1]);
        let t = Tensor::from_slice(&[52.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_052() {
        let p = Tensor::from_slice(&[52.0], vec![1]);
        let t = Tensor::from_slice(&[53.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_053() {
        let p = Tensor::from_slice(&[53.0], vec![1]);
        let t = Tensor::from_slice(&[54.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_054() {
        let p = Tensor::from_slice(&[54.0], vec![1]);
        let t = Tensor::from_slice(&[55.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_055() {
        let p = Tensor::from_slice(&[55.0], vec![1]);
        let t = Tensor::from_slice(&[56.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_056() {
        let p = Tensor::from_slice(&[56.0], vec![1]);
        let t = Tensor::from_slice(&[57.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_057() {
        let p = Tensor::from_slice(&[57.0], vec![1]);
        let t = Tensor::from_slice(&[58.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_058() {
        let p = Tensor::from_slice(&[58.0], vec![1]);
        let t = Tensor::from_slice(&[59.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_059() {
        let p = Tensor::from_slice(&[59.0], vec![1]);
        let t = Tensor::from_slice(&[60.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_060() {
        let p = Tensor::from_slice(&[60.0], vec![1]);
        let t = Tensor::from_slice(&[61.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_061() {
        let p = Tensor::from_slice(&[61.0], vec![1]);
        let t = Tensor::from_slice(&[62.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_062() {
        let p = Tensor::from_slice(&[62.0], vec![1]);
        let t = Tensor::from_slice(&[63.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_063() {
        let p = Tensor::from_slice(&[63.0], vec![1]);
        let t = Tensor::from_slice(&[64.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_064() {
        let p = Tensor::from_slice(&[64.0], vec![1]);
        let t = Tensor::from_slice(&[65.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_065() {
        let p = Tensor::from_slice(&[65.0], vec![1]);
        let t = Tensor::from_slice(&[66.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_066() {
        let p = Tensor::from_slice(&[66.0], vec![1]);
        let t = Tensor::from_slice(&[67.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_067() {
        let p = Tensor::from_slice(&[67.0], vec![1]);
        let t = Tensor::from_slice(&[68.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_068() {
        let p = Tensor::from_slice(&[68.0], vec![1]);
        let t = Tensor::from_slice(&[69.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_069() {
        let p = Tensor::from_slice(&[69.0], vec![1]);
        let t = Tensor::from_slice(&[70.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_070() {
        let p = Tensor::from_slice(&[70.0], vec![1]);
        let t = Tensor::from_slice(&[71.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_071() {
        let p = Tensor::from_slice(&[71.0], vec![1]);
        let t = Tensor::from_slice(&[72.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_072() {
        let p = Tensor::from_slice(&[72.0], vec![1]);
        let t = Tensor::from_slice(&[73.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_073() {
        let p = Tensor::from_slice(&[73.0], vec![1]);
        let t = Tensor::from_slice(&[74.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_074() {
        let p = Tensor::from_slice(&[74.0], vec![1]);
        let t = Tensor::from_slice(&[75.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_075() {
        let p = Tensor::from_slice(&[75.0], vec![1]);
        let t = Tensor::from_slice(&[76.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_076() {
        let p = Tensor::from_slice(&[76.0], vec![1]);
        let t = Tensor::from_slice(&[77.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_077() {
        let p = Tensor::from_slice(&[77.0], vec![1]);
        let t = Tensor::from_slice(&[78.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_078() {
        let p = Tensor::from_slice(&[78.0], vec![1]);
        let t = Tensor::from_slice(&[79.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_079() {
        let p = Tensor::from_slice(&[79.0], vec![1]);
        let t = Tensor::from_slice(&[80.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_080() {
        let p = Tensor::from_slice(&[80.0], vec![1]);
        let t = Tensor::from_slice(&[81.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_081() {
        let p = Tensor::from_slice(&[81.0], vec![1]);
        let t = Tensor::from_slice(&[82.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_082() {
        let p = Tensor::from_slice(&[82.0], vec![1]);
        let t = Tensor::from_slice(&[83.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_083() {
        let p = Tensor::from_slice(&[83.0], vec![1]);
        let t = Tensor::from_slice(&[84.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_084() {
        let p = Tensor::from_slice(&[84.0], vec![1]);
        let t = Tensor::from_slice(&[85.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_085() {
        let p = Tensor::from_slice(&[85.0], vec![1]);
        let t = Tensor::from_slice(&[86.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_086() {
        let p = Tensor::from_slice(&[86.0], vec![1]);
        let t = Tensor::from_slice(&[87.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_087() {
        let p = Tensor::from_slice(&[87.0], vec![1]);
        let t = Tensor::from_slice(&[88.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_088() {
        let p = Tensor::from_slice(&[88.0], vec![1]);
        let t = Tensor::from_slice(&[89.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_089() {
        let p = Tensor::from_slice(&[89.0], vec![1]);
        let t = Tensor::from_slice(&[90.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_090() {
        let p = Tensor::from_slice(&[90.0], vec![1]);
        let t = Tensor::from_slice(&[91.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_091() {
        let p = Tensor::from_slice(&[91.0], vec![1]);
        let t = Tensor::from_slice(&[92.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_092() {
        let p = Tensor::from_slice(&[92.0], vec![1]);
        let t = Tensor::from_slice(&[93.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_093() {
        let p = Tensor::from_slice(&[93.0], vec![1]);
        let t = Tensor::from_slice(&[94.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_094() {
        let p = Tensor::from_slice(&[94.0], vec![1]);
        let t = Tensor::from_slice(&[95.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_095() {
        let p = Tensor::from_slice(&[95.0], vec![1]);
        let t = Tensor::from_slice(&[96.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_096() {
        let p = Tensor::from_slice(&[96.0], vec![1]);
        let t = Tensor::from_slice(&[97.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_097() {
        let p = Tensor::from_slice(&[97.0], vec![1]);
        let t = Tensor::from_slice(&[98.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_098() {
        let p = Tensor::from_slice(&[98.0], vec![1]);
        let t = Tensor::from_slice(&[99.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_099() {
        let p = Tensor::from_slice(&[99.0], vec![1]);
        let t = Tensor::from_slice(&[100.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_100() {
        let p = Tensor::from_slice(&[100.0], vec![1]);
        let t = Tensor::from_slice(&[101.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_101() {
        let p = Tensor::from_slice(&[101.0], vec![1]);
        let t = Tensor::from_slice(&[102.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_102() {
        let p = Tensor::from_slice(&[102.0], vec![1]);
        let t = Tensor::from_slice(&[103.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_103() {
        let p = Tensor::from_slice(&[103.0], vec![1]);
        let t = Tensor::from_slice(&[104.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_104() {
        let p = Tensor::from_slice(&[104.0], vec![1]);
        let t = Tensor::from_slice(&[105.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_105() {
        let p = Tensor::from_slice(&[105.0], vec![1]);
        let t = Tensor::from_slice(&[106.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_106() {
        let p = Tensor::from_slice(&[106.0], vec![1]);
        let t = Tensor::from_slice(&[107.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_107() {
        let p = Tensor::from_slice(&[107.0], vec![1]);
        let t = Tensor::from_slice(&[108.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_108() {
        let p = Tensor::from_slice(&[108.0], vec![1]);
        let t = Tensor::from_slice(&[109.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_109() {
        let p = Tensor::from_slice(&[109.0], vec![1]);
        let t = Tensor::from_slice(&[110.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_110() {
        let p = Tensor::from_slice(&[110.0], vec![1]);
        let t = Tensor::from_slice(&[111.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_111() {
        let p = Tensor::from_slice(&[111.0], vec![1]);
        let t = Tensor::from_slice(&[112.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_112() {
        let p = Tensor::from_slice(&[112.0], vec![1]);
        let t = Tensor::from_slice(&[113.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_113() {
        let p = Tensor::from_slice(&[113.0], vec![1]);
        let t = Tensor::from_slice(&[114.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_114() {
        let p = Tensor::from_slice(&[114.0], vec![1]);
        let t = Tensor::from_slice(&[115.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_115() {
        let p = Tensor::from_slice(&[115.0], vec![1]);
        let t = Tensor::from_slice(&[116.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_116() {
        let p = Tensor::from_slice(&[116.0], vec![1]);
        let t = Tensor::from_slice(&[117.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_117() {
        let p = Tensor::from_slice(&[117.0], vec![1]);
        let t = Tensor::from_slice(&[118.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_118() {
        let p = Tensor::from_slice(&[118.0], vec![1]);
        let t = Tensor::from_slice(&[119.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_119() {
        let p = Tensor::from_slice(&[119.0], vec![1]);
        let t = Tensor::from_slice(&[120.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_120() {
        let p = Tensor::from_slice(&[120.0], vec![1]);
        let t = Tensor::from_slice(&[121.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_121() {
        let p = Tensor::from_slice(&[121.0], vec![1]);
        let t = Tensor::from_slice(&[122.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_122() {
        let p = Tensor::from_slice(&[122.0], vec![1]);
        let t = Tensor::from_slice(&[123.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_123() {
        let p = Tensor::from_slice(&[123.0], vec![1]);
        let t = Tensor::from_slice(&[124.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_124() {
        let p = Tensor::from_slice(&[124.0], vec![1]);
        let t = Tensor::from_slice(&[125.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_125() {
        let p = Tensor::from_slice(&[125.0], vec![1]);
        let t = Tensor::from_slice(&[126.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_126() {
        let p = Tensor::from_slice(&[126.0], vec![1]);
        let t = Tensor::from_slice(&[127.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_127() {
        let p = Tensor::from_slice(&[127.0], vec![1]);
        let t = Tensor::from_slice(&[128.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_128() {
        let p = Tensor::from_slice(&[128.0], vec![1]);
        let t = Tensor::from_slice(&[129.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_129() {
        let p = Tensor::from_slice(&[129.0], vec![1]);
        let t = Tensor::from_slice(&[130.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_130() {
        let p = Tensor::from_slice(&[130.0], vec![1]);
        let t = Tensor::from_slice(&[131.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_131() {
        let p = Tensor::from_slice(&[131.0], vec![1]);
        let t = Tensor::from_slice(&[132.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_132() {
        let p = Tensor::from_slice(&[132.0], vec![1]);
        let t = Tensor::from_slice(&[133.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_133() {
        let p = Tensor::from_slice(&[133.0], vec![1]);
        let t = Tensor::from_slice(&[134.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_134() {
        let p = Tensor::from_slice(&[134.0], vec![1]);
        let t = Tensor::from_slice(&[135.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_135() {
        let p = Tensor::from_slice(&[135.0], vec![1]);
        let t = Tensor::from_slice(&[136.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_136() {
        let p = Tensor::from_slice(&[136.0], vec![1]);
        let t = Tensor::from_slice(&[137.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_137() {
        let p = Tensor::from_slice(&[137.0], vec![1]);
        let t = Tensor::from_slice(&[138.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_138() {
        let p = Tensor::from_slice(&[138.0], vec![1]);
        let t = Tensor::from_slice(&[139.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_139() {
        let p = Tensor::from_slice(&[139.0], vec![1]);
        let t = Tensor::from_slice(&[140.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_140() {
        let p = Tensor::from_slice(&[140.0], vec![1]);
        let t = Tensor::from_slice(&[141.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_141() {
        let p = Tensor::from_slice(&[141.0], vec![1]);
        let t = Tensor::from_slice(&[142.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_142() {
        let p = Tensor::from_slice(&[142.0], vec![1]);
        let t = Tensor::from_slice(&[143.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_143() {
        let p = Tensor::from_slice(&[143.0], vec![1]);
        let t = Tensor::from_slice(&[144.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_144() {
        let p = Tensor::from_slice(&[144.0], vec![1]);
        let t = Tensor::from_slice(&[145.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_145() {
        let p = Tensor::from_slice(&[145.0], vec![1]);
        let t = Tensor::from_slice(&[146.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_146() {
        let p = Tensor::from_slice(&[146.0], vec![1]);
        let t = Tensor::from_slice(&[147.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_147() {
        let p = Tensor::from_slice(&[147.0], vec![1]);
        let t = Tensor::from_slice(&[148.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_148() {
        let p = Tensor::from_slice(&[148.0], vec![1]);
        let t = Tensor::from_slice(&[149.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_149() {
        let p = Tensor::from_slice(&[149.0], vec![1]);
        let t = Tensor::from_slice(&[150.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_150() {
        let p = Tensor::from_slice(&[150.0], vec![1]);
        let t = Tensor::from_slice(&[151.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_151() {
        let p = Tensor::from_slice(&[151.0], vec![1]);
        let t = Tensor::from_slice(&[152.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_152() {
        let p = Tensor::from_slice(&[152.0], vec![1]);
        let t = Tensor::from_slice(&[153.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_153() {
        let p = Tensor::from_slice(&[153.0], vec![1]);
        let t = Tensor::from_slice(&[154.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_154() {
        let p = Tensor::from_slice(&[154.0], vec![1]);
        let t = Tensor::from_slice(&[155.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_155() {
        let p = Tensor::from_slice(&[155.0], vec![1]);
        let t = Tensor::from_slice(&[156.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_156() {
        let p = Tensor::from_slice(&[156.0], vec![1]);
        let t = Tensor::from_slice(&[157.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_157() {
        let p = Tensor::from_slice(&[157.0], vec![1]);
        let t = Tensor::from_slice(&[158.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_158() {
        let p = Tensor::from_slice(&[158.0], vec![1]);
        let t = Tensor::from_slice(&[159.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_159() {
        let p = Tensor::from_slice(&[159.0], vec![1]);
        let t = Tensor::from_slice(&[160.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_160() {
        let p = Tensor::from_slice(&[160.0], vec![1]);
        let t = Tensor::from_slice(&[161.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_161() {
        let p = Tensor::from_slice(&[161.0], vec![1]);
        let t = Tensor::from_slice(&[162.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_162() {
        let p = Tensor::from_slice(&[162.0], vec![1]);
        let t = Tensor::from_slice(&[163.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_163() {
        let p = Tensor::from_slice(&[163.0], vec![1]);
        let t = Tensor::from_slice(&[164.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_164() {
        let p = Tensor::from_slice(&[164.0], vec![1]);
        let t = Tensor::from_slice(&[165.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_165() {
        let p = Tensor::from_slice(&[165.0], vec![1]);
        let t = Tensor::from_slice(&[166.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_166() {
        let p = Tensor::from_slice(&[166.0], vec![1]);
        let t = Tensor::from_slice(&[167.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_167() {
        let p = Tensor::from_slice(&[167.0], vec![1]);
        let t = Tensor::from_slice(&[168.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_168() {
        let p = Tensor::from_slice(&[168.0], vec![1]);
        let t = Tensor::from_slice(&[169.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_169() {
        let p = Tensor::from_slice(&[169.0], vec![1]);
        let t = Tensor::from_slice(&[170.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_170() {
        let p = Tensor::from_slice(&[170.0], vec![1]);
        let t = Tensor::from_slice(&[171.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_171() {
        let p = Tensor::from_slice(&[171.0], vec![1]);
        let t = Tensor::from_slice(&[172.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_172() {
        let p = Tensor::from_slice(&[172.0], vec![1]);
        let t = Tensor::from_slice(&[173.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_173() {
        let p = Tensor::from_slice(&[173.0], vec![1]);
        let t = Tensor::from_slice(&[174.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_174() {
        let p = Tensor::from_slice(&[174.0], vec![1]);
        let t = Tensor::from_slice(&[175.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_175() {
        let p = Tensor::from_slice(&[175.0], vec![1]);
        let t = Tensor::from_slice(&[176.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_176() {
        let p = Tensor::from_slice(&[176.0], vec![1]);
        let t = Tensor::from_slice(&[177.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_177() {
        let p = Tensor::from_slice(&[177.0], vec![1]);
        let t = Tensor::from_slice(&[178.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_178() {
        let p = Tensor::from_slice(&[178.0], vec![1]);
        let t = Tensor::from_slice(&[179.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_179() {
        let p = Tensor::from_slice(&[179.0], vec![1]);
        let t = Tensor::from_slice(&[180.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_180() {
        let p = Tensor::from_slice(&[180.0], vec![1]);
        let t = Tensor::from_slice(&[181.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_181() {
        let p = Tensor::from_slice(&[181.0], vec![1]);
        let t = Tensor::from_slice(&[182.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_182() {
        let p = Tensor::from_slice(&[182.0], vec![1]);
        let t = Tensor::from_slice(&[183.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_183() {
        let p = Tensor::from_slice(&[183.0], vec![1]);
        let t = Tensor::from_slice(&[184.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_184() {
        let p = Tensor::from_slice(&[184.0], vec![1]);
        let t = Tensor::from_slice(&[185.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_185() {
        let p = Tensor::from_slice(&[185.0], vec![1]);
        let t = Tensor::from_slice(&[186.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_186() {
        let p = Tensor::from_slice(&[186.0], vec![1]);
        let t = Tensor::from_slice(&[187.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_187() {
        let p = Tensor::from_slice(&[187.0], vec![1]);
        let t = Tensor::from_slice(&[188.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_188() {
        let p = Tensor::from_slice(&[188.0], vec![1]);
        let t = Tensor::from_slice(&[189.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_189() {
        let p = Tensor::from_slice(&[189.0], vec![1]);
        let t = Tensor::from_slice(&[190.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_190() {
        let p = Tensor::from_slice(&[190.0], vec![1]);
        let t = Tensor::from_slice(&[191.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_191() {
        let p = Tensor::from_slice(&[191.0], vec![1]);
        let t = Tensor::from_slice(&[192.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_192() {
        let p = Tensor::from_slice(&[192.0], vec![1]);
        let t = Tensor::from_slice(&[193.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_193() {
        let p = Tensor::from_slice(&[193.0], vec![1]);
        let t = Tensor::from_slice(&[194.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_194() {
        let p = Tensor::from_slice(&[194.0], vec![1]);
        let t = Tensor::from_slice(&[195.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_195() {
        let p = Tensor::from_slice(&[195.0], vec![1]);
        let t = Tensor::from_slice(&[196.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_196() {
        let p = Tensor::from_slice(&[196.0], vec![1]);
        let t = Tensor::from_slice(&[197.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_197() {
        let p = Tensor::from_slice(&[197.0], vec![1]);
        let t = Tensor::from_slice(&[198.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_198() {
        let p = Tensor::from_slice(&[198.0], vec![1]);
        let t = Tensor::from_slice(&[199.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_199() {
        let p = Tensor::from_slice(&[199.0], vec![1]);
        let t = Tensor::from_slice(&[200.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_200() {
        let p = Tensor::from_slice(&[200.0], vec![1]);
        let t = Tensor::from_slice(&[201.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_201() {
        let p = Tensor::from_slice(&[201.0], vec![1]);
        let t = Tensor::from_slice(&[202.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_202() {
        let p = Tensor::from_slice(&[202.0], vec![1]);
        let t = Tensor::from_slice(&[203.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_203() {
        let p = Tensor::from_slice(&[203.0], vec![1]);
        let t = Tensor::from_slice(&[204.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_204() {
        let p = Tensor::from_slice(&[204.0], vec![1]);
        let t = Tensor::from_slice(&[205.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_205() {
        let p = Tensor::from_slice(&[205.0], vec![1]);
        let t = Tensor::from_slice(&[206.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_206() {
        let p = Tensor::from_slice(&[206.0], vec![1]);
        let t = Tensor::from_slice(&[207.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_207() {
        let p = Tensor::from_slice(&[207.0], vec![1]);
        let t = Tensor::from_slice(&[208.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_208() {
        let p = Tensor::from_slice(&[208.0], vec![1]);
        let t = Tensor::from_slice(&[209.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_209() {
        let p = Tensor::from_slice(&[209.0], vec![1]);
        let t = Tensor::from_slice(&[210.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_210() {
        let p = Tensor::from_slice(&[210.0], vec![1]);
        let t = Tensor::from_slice(&[211.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_211() {
        let p = Tensor::from_slice(&[211.0], vec![1]);
        let t = Tensor::from_slice(&[212.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_212() {
        let p = Tensor::from_slice(&[212.0], vec![1]);
        let t = Tensor::from_slice(&[213.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_213() {
        let p = Tensor::from_slice(&[213.0], vec![1]);
        let t = Tensor::from_slice(&[214.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_214() {
        let p = Tensor::from_slice(&[214.0], vec![1]);
        let t = Tensor::from_slice(&[215.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_215() {
        let p = Tensor::from_slice(&[215.0], vec![1]);
        let t = Tensor::from_slice(&[216.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_216() {
        let p = Tensor::from_slice(&[216.0], vec![1]);
        let t = Tensor::from_slice(&[217.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_217() {
        let p = Tensor::from_slice(&[217.0], vec![1]);
        let t = Tensor::from_slice(&[218.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_218() {
        let p = Tensor::from_slice(&[218.0], vec![1]);
        let t = Tensor::from_slice(&[219.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_219() {
        let p = Tensor::from_slice(&[219.0], vec![1]);
        let t = Tensor::from_slice(&[220.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_220() {
        let p = Tensor::from_slice(&[220.0], vec![1]);
        let t = Tensor::from_slice(&[221.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_221() {
        let p = Tensor::from_slice(&[221.0], vec![1]);
        let t = Tensor::from_slice(&[222.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_222() {
        let p = Tensor::from_slice(&[222.0], vec![1]);
        let t = Tensor::from_slice(&[223.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_223() {
        let p = Tensor::from_slice(&[223.0], vec![1]);
        let t = Tensor::from_slice(&[224.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_224() {
        let p = Tensor::from_slice(&[224.0], vec![1]);
        let t = Tensor::from_slice(&[225.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_225() {
        let p = Tensor::from_slice(&[225.0], vec![1]);
        let t = Tensor::from_slice(&[226.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_226() {
        let p = Tensor::from_slice(&[226.0], vec![1]);
        let t = Tensor::from_slice(&[227.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_227() {
        let p = Tensor::from_slice(&[227.0], vec![1]);
        let t = Tensor::from_slice(&[228.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_228() {
        let p = Tensor::from_slice(&[228.0], vec![1]);
        let t = Tensor::from_slice(&[229.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_229() {
        let p = Tensor::from_slice(&[229.0], vec![1]);
        let t = Tensor::from_slice(&[230.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_230() {
        let p = Tensor::from_slice(&[230.0], vec![1]);
        let t = Tensor::from_slice(&[231.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_231() {
        let p = Tensor::from_slice(&[231.0], vec![1]);
        let t = Tensor::from_slice(&[232.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_232() {
        let p = Tensor::from_slice(&[232.0], vec![1]);
        let t = Tensor::from_slice(&[233.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_233() {
        let p = Tensor::from_slice(&[233.0], vec![1]);
        let t = Tensor::from_slice(&[234.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_234() {
        let p = Tensor::from_slice(&[234.0], vec![1]);
        let t = Tensor::from_slice(&[235.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_235() {
        let p = Tensor::from_slice(&[235.0], vec![1]);
        let t = Tensor::from_slice(&[236.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_236() {
        let p = Tensor::from_slice(&[236.0], vec![1]);
        let t = Tensor::from_slice(&[237.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_237() {
        let p = Tensor::from_slice(&[237.0], vec![1]);
        let t = Tensor::from_slice(&[238.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_238() {
        let p = Tensor::from_slice(&[238.0], vec![1]);
        let t = Tensor::from_slice(&[239.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_239() {
        let p = Tensor::from_slice(&[239.0], vec![1]);
        let t = Tensor::from_slice(&[240.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_240() {
        let p = Tensor::from_slice(&[240.0], vec![1]);
        let t = Tensor::from_slice(&[241.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_241() {
        let p = Tensor::from_slice(&[241.0], vec![1]);
        let t = Tensor::from_slice(&[242.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_242() {
        let p = Tensor::from_slice(&[242.0], vec![1]);
        let t = Tensor::from_slice(&[243.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_243() {
        let p = Tensor::from_slice(&[243.0], vec![1]);
        let t = Tensor::from_slice(&[244.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_244() {
        let p = Tensor::from_slice(&[244.0], vec![1]);
        let t = Tensor::from_slice(&[245.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_245() {
        let p = Tensor::from_slice(&[245.0], vec![1]);
        let t = Tensor::from_slice(&[246.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_246() {
        let p = Tensor::from_slice(&[246.0], vec![1]);
        let t = Tensor::from_slice(&[247.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_247() {
        let p = Tensor::from_slice(&[247.0], vec![1]);
        let t = Tensor::from_slice(&[248.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_248() {
        let p = Tensor::from_slice(&[248.0], vec![1]);
        let t = Tensor::from_slice(&[249.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_249() {
        let p = Tensor::from_slice(&[249.0], vec![1]);
        let t = Tensor::from_slice(&[250.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_250() {
        let p = Tensor::from_slice(&[250.0], vec![1]);
        let t = Tensor::from_slice(&[251.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_251() {
        let p = Tensor::from_slice(&[251.0], vec![1]);
        let t = Tensor::from_slice(&[252.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_252() {
        let p = Tensor::from_slice(&[252.0], vec![1]);
        let t = Tensor::from_slice(&[253.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_253() {
        let p = Tensor::from_slice(&[253.0], vec![1]);
        let t = Tensor::from_slice(&[254.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_254() {
        let p = Tensor::from_slice(&[254.0], vec![1]);
        let t = Tensor::from_slice(&[255.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_255() {
        let p = Tensor::from_slice(&[255.0], vec![1]);
        let t = Tensor::from_slice(&[256.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_256() {
        let p = Tensor::from_slice(&[256.0], vec![1]);
        let t = Tensor::from_slice(&[257.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_257() {
        let p = Tensor::from_slice(&[257.0], vec![1]);
        let t = Tensor::from_slice(&[258.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_258() {
        let p = Tensor::from_slice(&[258.0], vec![1]);
        let t = Tensor::from_slice(&[259.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_259() {
        let p = Tensor::from_slice(&[259.0], vec![1]);
        let t = Tensor::from_slice(&[260.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_260() {
        let p = Tensor::from_slice(&[260.0], vec![1]);
        let t = Tensor::from_slice(&[261.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_261() {
        let p = Tensor::from_slice(&[261.0], vec![1]);
        let t = Tensor::from_slice(&[262.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_262() {
        let p = Tensor::from_slice(&[262.0], vec![1]);
        let t = Tensor::from_slice(&[263.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_263() {
        let p = Tensor::from_slice(&[263.0], vec![1]);
        let t = Tensor::from_slice(&[264.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_264() {
        let p = Tensor::from_slice(&[264.0], vec![1]);
        let t = Tensor::from_slice(&[265.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_265() {
        let p = Tensor::from_slice(&[265.0], vec![1]);
        let t = Tensor::from_slice(&[266.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_266() {
        let p = Tensor::from_slice(&[266.0], vec![1]);
        let t = Tensor::from_slice(&[267.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_267() {
        let p = Tensor::from_slice(&[267.0], vec![1]);
        let t = Tensor::from_slice(&[268.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_268() {
        let p = Tensor::from_slice(&[268.0], vec![1]);
        let t = Tensor::from_slice(&[269.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_269() {
        let p = Tensor::from_slice(&[269.0], vec![1]);
        let t = Tensor::from_slice(&[270.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_270() {
        let p = Tensor::from_slice(&[270.0], vec![1]);
        let t = Tensor::from_slice(&[271.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_271() {
        let p = Tensor::from_slice(&[271.0], vec![1]);
        let t = Tensor::from_slice(&[272.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_272() {
        let p = Tensor::from_slice(&[272.0], vec![1]);
        let t = Tensor::from_slice(&[273.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_273() {
        let p = Tensor::from_slice(&[273.0], vec![1]);
        let t = Tensor::from_slice(&[274.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_274() {
        let p = Tensor::from_slice(&[274.0], vec![1]);
        let t = Tensor::from_slice(&[275.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_275() {
        let p = Tensor::from_slice(&[275.0], vec![1]);
        let t = Tensor::from_slice(&[276.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_276() {
        let p = Tensor::from_slice(&[276.0], vec![1]);
        let t = Tensor::from_slice(&[277.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_277() {
        let p = Tensor::from_slice(&[277.0], vec![1]);
        let t = Tensor::from_slice(&[278.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_278() {
        let p = Tensor::from_slice(&[278.0], vec![1]);
        let t = Tensor::from_slice(&[279.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_279() {
        let p = Tensor::from_slice(&[279.0], vec![1]);
        let t = Tensor::from_slice(&[280.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_280() {
        let p = Tensor::from_slice(&[280.0], vec![1]);
        let t = Tensor::from_slice(&[281.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_281() {
        let p = Tensor::from_slice(&[281.0], vec![1]);
        let t = Tensor::from_slice(&[282.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_282() {
        let p = Tensor::from_slice(&[282.0], vec![1]);
        let t = Tensor::from_slice(&[283.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_283() {
        let p = Tensor::from_slice(&[283.0], vec![1]);
        let t = Tensor::from_slice(&[284.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_284() {
        let p = Tensor::from_slice(&[284.0], vec![1]);
        let t = Tensor::from_slice(&[285.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_285() {
        let p = Tensor::from_slice(&[285.0], vec![1]);
        let t = Tensor::from_slice(&[286.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_286() {
        let p = Tensor::from_slice(&[286.0], vec![1]);
        let t = Tensor::from_slice(&[287.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_287() {
        let p = Tensor::from_slice(&[287.0], vec![1]);
        let t = Tensor::from_slice(&[288.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_288() {
        let p = Tensor::from_slice(&[288.0], vec![1]);
        let t = Tensor::from_slice(&[289.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_289() {
        let p = Tensor::from_slice(&[289.0], vec![1]);
        let t = Tensor::from_slice(&[290.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_290() {
        let p = Tensor::from_slice(&[290.0], vec![1]);
        let t = Tensor::from_slice(&[291.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_291() {
        let p = Tensor::from_slice(&[291.0], vec![1]);
        let t = Tensor::from_slice(&[292.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_292() {
        let p = Tensor::from_slice(&[292.0], vec![1]);
        let t = Tensor::from_slice(&[293.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_293() {
        let p = Tensor::from_slice(&[293.0], vec![1]);
        let t = Tensor::from_slice(&[294.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_294() {
        let p = Tensor::from_slice(&[294.0], vec![1]);
        let t = Tensor::from_slice(&[295.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_295() {
        let p = Tensor::from_slice(&[295.0], vec![1]);
        let t = Tensor::from_slice(&[296.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_296() {
        let p = Tensor::from_slice(&[296.0], vec![1]);
        let t = Tensor::from_slice(&[297.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_297() {
        let p = Tensor::from_slice(&[297.0], vec![1]);
        let t = Tensor::from_slice(&[298.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_298() {
        let p = Tensor::from_slice(&[298.0], vec![1]);
        let t = Tensor::from_slice(&[299.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_299() {
        let p = Tensor::from_slice(&[299.0], vec![1]);
        let t = Tensor::from_slice(&[300.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_300() {
        let p = Tensor::from_slice(&[300.0], vec![1]);
        let t = Tensor::from_slice(&[301.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_301() {
        let p = Tensor::from_slice(&[301.0], vec![1]);
        let t = Tensor::from_slice(&[302.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_302() {
        let p = Tensor::from_slice(&[302.0], vec![1]);
        let t = Tensor::from_slice(&[303.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_303() {
        let p = Tensor::from_slice(&[303.0], vec![1]);
        let t = Tensor::from_slice(&[304.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_304() {
        let p = Tensor::from_slice(&[304.0], vec![1]);
        let t = Tensor::from_slice(&[305.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_305() {
        let p = Tensor::from_slice(&[305.0], vec![1]);
        let t = Tensor::from_slice(&[306.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_306() {
        let p = Tensor::from_slice(&[306.0], vec![1]);
        let t = Tensor::from_slice(&[307.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_307() {
        let p = Tensor::from_slice(&[307.0], vec![1]);
        let t = Tensor::from_slice(&[308.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_308() {
        let p = Tensor::from_slice(&[308.0], vec![1]);
        let t = Tensor::from_slice(&[309.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_309() {
        let p = Tensor::from_slice(&[309.0], vec![1]);
        let t = Tensor::from_slice(&[310.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_310() {
        let p = Tensor::from_slice(&[310.0], vec![1]);
        let t = Tensor::from_slice(&[311.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_311() {
        let p = Tensor::from_slice(&[311.0], vec![1]);
        let t = Tensor::from_slice(&[312.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_312() {
        let p = Tensor::from_slice(&[312.0], vec![1]);
        let t = Tensor::from_slice(&[313.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_313() {
        let p = Tensor::from_slice(&[313.0], vec![1]);
        let t = Tensor::from_slice(&[314.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_314() {
        let p = Tensor::from_slice(&[314.0], vec![1]);
        let t = Tensor::from_slice(&[315.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_315() {
        let p = Tensor::from_slice(&[315.0], vec![1]);
        let t = Tensor::from_slice(&[316.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_316() {
        let p = Tensor::from_slice(&[316.0], vec![1]);
        let t = Tensor::from_slice(&[317.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_317() {
        let p = Tensor::from_slice(&[317.0], vec![1]);
        let t = Tensor::from_slice(&[318.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_318() {
        let p = Tensor::from_slice(&[318.0], vec![1]);
        let t = Tensor::from_slice(&[319.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_319() {
        let p = Tensor::from_slice(&[319.0], vec![1]);
        let t = Tensor::from_slice(&[320.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_320() {
        let p = Tensor::from_slice(&[320.0], vec![1]);
        let t = Tensor::from_slice(&[321.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_321() {
        let p = Tensor::from_slice(&[321.0], vec![1]);
        let t = Tensor::from_slice(&[322.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_322() {
        let p = Tensor::from_slice(&[322.0], vec![1]);
        let t = Tensor::from_slice(&[323.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_323() {
        let p = Tensor::from_slice(&[323.0], vec![1]);
        let t = Tensor::from_slice(&[324.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_324() {
        let p = Tensor::from_slice(&[324.0], vec![1]);
        let t = Tensor::from_slice(&[325.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_325() {
        let p = Tensor::from_slice(&[325.0], vec![1]);
        let t = Tensor::from_slice(&[326.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_326() {
        let p = Tensor::from_slice(&[326.0], vec![1]);
        let t = Tensor::from_slice(&[327.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_327() {
        let p = Tensor::from_slice(&[327.0], vec![1]);
        let t = Tensor::from_slice(&[328.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_328() {
        let p = Tensor::from_slice(&[328.0], vec![1]);
        let t = Tensor::from_slice(&[329.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_329() {
        let p = Tensor::from_slice(&[329.0], vec![1]);
        let t = Tensor::from_slice(&[330.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_330() {
        let p = Tensor::from_slice(&[330.0], vec![1]);
        let t = Tensor::from_slice(&[331.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_331() {
        let p = Tensor::from_slice(&[331.0], vec![1]);
        let t = Tensor::from_slice(&[332.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_332() {
        let p = Tensor::from_slice(&[332.0], vec![1]);
        let t = Tensor::from_slice(&[333.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_333() {
        let p = Tensor::from_slice(&[333.0], vec![1]);
        let t = Tensor::from_slice(&[334.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_334() {
        let p = Tensor::from_slice(&[334.0], vec![1]);
        let t = Tensor::from_slice(&[335.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_335() {
        let p = Tensor::from_slice(&[335.0], vec![1]);
        let t = Tensor::from_slice(&[336.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_336() {
        let p = Tensor::from_slice(&[336.0], vec![1]);
        let t = Tensor::from_slice(&[337.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_337() {
        let p = Tensor::from_slice(&[337.0], vec![1]);
        let t = Tensor::from_slice(&[338.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_338() {
        let p = Tensor::from_slice(&[338.0], vec![1]);
        let t = Tensor::from_slice(&[339.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_339() {
        let p = Tensor::from_slice(&[339.0], vec![1]);
        let t = Tensor::from_slice(&[340.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_340() {
        let p = Tensor::from_slice(&[340.0], vec![1]);
        let t = Tensor::from_slice(&[341.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_341() {
        let p = Tensor::from_slice(&[341.0], vec![1]);
        let t = Tensor::from_slice(&[342.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_342() {
        let p = Tensor::from_slice(&[342.0], vec![1]);
        let t = Tensor::from_slice(&[343.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_343() {
        let p = Tensor::from_slice(&[343.0], vec![1]);
        let t = Tensor::from_slice(&[344.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_344() {
        let p = Tensor::from_slice(&[344.0], vec![1]);
        let t = Tensor::from_slice(&[345.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_345() {
        let p = Tensor::from_slice(&[345.0], vec![1]);
        let t = Tensor::from_slice(&[346.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_346() {
        let p = Tensor::from_slice(&[346.0], vec![1]);
        let t = Tensor::from_slice(&[347.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_347() {
        let p = Tensor::from_slice(&[347.0], vec![1]);
        let t = Tensor::from_slice(&[348.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_348() {
        let p = Tensor::from_slice(&[348.0], vec![1]);
        let t = Tensor::from_slice(&[349.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_349() {
        let p = Tensor::from_slice(&[349.0], vec![1]);
        let t = Tensor::from_slice(&[350.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_350() {
        let p = Tensor::from_slice(&[350.0], vec![1]);
        let t = Tensor::from_slice(&[351.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_351() {
        let p = Tensor::from_slice(&[351.0], vec![1]);
        let t = Tensor::from_slice(&[352.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_352() {
        let p = Tensor::from_slice(&[352.0], vec![1]);
        let t = Tensor::from_slice(&[353.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_353() {
        let p = Tensor::from_slice(&[353.0], vec![1]);
        let t = Tensor::from_slice(&[354.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_354() {
        let p = Tensor::from_slice(&[354.0], vec![1]);
        let t = Tensor::from_slice(&[355.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_355() {
        let p = Tensor::from_slice(&[355.0], vec![1]);
        let t = Tensor::from_slice(&[356.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_356() {
        let p = Tensor::from_slice(&[356.0], vec![1]);
        let t = Tensor::from_slice(&[357.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_357() {
        let p = Tensor::from_slice(&[357.0], vec![1]);
        let t = Tensor::from_slice(&[358.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_358() {
        let p = Tensor::from_slice(&[358.0], vec![1]);
        let t = Tensor::from_slice(&[359.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_359() {
        let p = Tensor::from_slice(&[359.0], vec![1]);
        let t = Tensor::from_slice(&[360.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_360() {
        let p = Tensor::from_slice(&[360.0], vec![1]);
        let t = Tensor::from_slice(&[361.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_361() {
        let p = Tensor::from_slice(&[361.0], vec![1]);
        let t = Tensor::from_slice(&[362.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_362() {
        let p = Tensor::from_slice(&[362.0], vec![1]);
        let t = Tensor::from_slice(&[363.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_363() {
        let p = Tensor::from_slice(&[363.0], vec![1]);
        let t = Tensor::from_slice(&[364.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_364() {
        let p = Tensor::from_slice(&[364.0], vec![1]);
        let t = Tensor::from_slice(&[365.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_365() {
        let p = Tensor::from_slice(&[365.0], vec![1]);
        let t = Tensor::from_slice(&[366.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_366() {
        let p = Tensor::from_slice(&[366.0], vec![1]);
        let t = Tensor::from_slice(&[367.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_367() {
        let p = Tensor::from_slice(&[367.0], vec![1]);
        let t = Tensor::from_slice(&[368.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_368() {
        let p = Tensor::from_slice(&[368.0], vec![1]);
        let t = Tensor::from_slice(&[369.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_369() {
        let p = Tensor::from_slice(&[369.0], vec![1]);
        let t = Tensor::from_slice(&[370.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_370() {
        let p = Tensor::from_slice(&[370.0], vec![1]);
        let t = Tensor::from_slice(&[371.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_371() {
        let p = Tensor::from_slice(&[371.0], vec![1]);
        let t = Tensor::from_slice(&[372.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_372() {
        let p = Tensor::from_slice(&[372.0], vec![1]);
        let t = Tensor::from_slice(&[373.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_373() {
        let p = Tensor::from_slice(&[373.0], vec![1]);
        let t = Tensor::from_slice(&[374.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_374() {
        let p = Tensor::from_slice(&[374.0], vec![1]);
        let t = Tensor::from_slice(&[375.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_375() {
        let p = Tensor::from_slice(&[375.0], vec![1]);
        let t = Tensor::from_slice(&[376.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_376() {
        let p = Tensor::from_slice(&[376.0], vec![1]);
        let t = Tensor::from_slice(&[377.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_377() {
        let p = Tensor::from_slice(&[377.0], vec![1]);
        let t = Tensor::from_slice(&[378.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_378() {
        let p = Tensor::from_slice(&[378.0], vec![1]);
        let t = Tensor::from_slice(&[379.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }

    #[test]
    fn test_neural_stress_case_379() {
        let p = Tensor::from_slice(&[379.0], vec![1]);
        let t = Tensor::from_slice(&[380.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }
}
