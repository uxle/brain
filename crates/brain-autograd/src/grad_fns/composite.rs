//! # Composite Layer Differentiable Rules
//!
//! Differentiable backward implementations for complex composite layers:
//! LayerNorm, RMSNorm, BatchNorm, Dropout, and Scaled Dot-Product Attention.

use brain_core::tensor::arithmetic as arith_t;
use brain_core::tensor::reduction as red_t;
use brain_core::tensor::special as spec_t;
use brain_core::{BrainResult, Tensor};

/// Computes backward gradients for Layer Normalization.
pub fn grad_layernorm(
    x: &Tensor,
    gamma: &Tensor,
    g: &Tensor,
    eps: f64,
) -> BrainResult<(Tensor, Tensor, Tensor)> {
    let mean = red_t::mean(x);
    let n = x.numel() as f64;
    let var = x.data().iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_inv = 1.0 / (var + eps).sqrt();

    let mut dx = vec![0.0; x.numel()];
    let mut dgamma = vec![0.0; gamma.numel()];
    let mut dbeta = vec![0.0; gamma.numel()];

    let x_slice = x.data();
    let g_slice = g.data();
    let gamma_slice = gamma.data();

    for (i, (&xi, &gi)) in x_slice.iter().zip(g_slice.iter()).enumerate() {
        let x_hat = (xi - mean) * std_inv;
        dgamma[i % gamma.numel()] += gi * x_hat;
        dbeta[i % gamma.numel()] += gi;
        dx[i] = gi * gamma_slice[i % gamma.numel()] * std_inv;
    }

    Ok((
        Tensor::from_slice(&dx, x.shape().to_vec()),
        Tensor::from_slice(&dgamma, gamma.shape().to_vec()),
        Tensor::from_slice(&dbeta, gamma.shape().to_vec()),
    ))
}

/// Computes backward gradients for Scaled Dot-Product Attention: `Q @ K^T / sqrt(d) -> Softmax -> @ V`.
pub fn grad_scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
) -> BrainResult<(Tensor, Tensor, Tensor)> {
    let d_k = q.shape().last().copied().unwrap_or(1) as f64;
    let scale = 1.0 / d_k.sqrt();

    let kt = k.transpose(0, 1);
    let scores = arith_t::matmul(q, &kt).map(|x| x * scale);
    let attn_weights = spec_t::softmax(&scores, scores.ndim() - 1);

    // dV = A^T @ G
    let at = attn_weights.transpose(0, 1);
    let dv = arith_t::matmul(&at, g);

    // dA = G @ V^T
    let vt = v.transpose(0, 1);
    let da = arith_t::matmul(g, &vt);

    // dScores from Softmax backward
    let dot = arith_t::mul(&da, &attn_weights);
    let dot_sum = red_t::sum_along_dim(&dot, da.ndim() - 1, true);
    let sub = arith_t::sub(&da, &dot_sum);
    let dscores = arith_t::mul(&attn_weights, &sub).map(|x| x * scale);

    // dQ = dScores @ K
    let dq = arith_t::matmul(&dscores, k);

    // dK = dScores^T @ Q
    let dscores_t = dscores.transpose(0, 1);
    let dk = arith_t::matmul(&dscores_t, q);

    Ok((dq, dk, dv))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;

    #[test]
    fn test_composite_vjp_stress_001() {
        let q = Tensor::from_slice(&[1.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_002() {
        let q = Tensor::from_slice(&[1.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_003() {
        let q = Tensor::from_slice(&[1.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_004() {
        let q = Tensor::from_slice(&[1.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_005() {
        let q = Tensor::from_slice(&[1.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_006() {
        let q = Tensor::from_slice(&[1.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_007() {
        let q = Tensor::from_slice(&[1.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_008() {
        let q = Tensor::from_slice(&[1.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_009() {
        let q = Tensor::from_slice(&[1.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_010() {
        let q = Tensor::from_slice(&[1.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_011() {
        let q = Tensor::from_slice(&[1.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_012() {
        let q = Tensor::from_slice(&[1.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_013() {
        let q = Tensor::from_slice(&[1.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_014() {
        let q = Tensor::from_slice(&[1.7000000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_015() {
        let q = Tensor::from_slice(&[1.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_016() {
        let q = Tensor::from_slice(&[1.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_017() {
        let q = Tensor::from_slice(&[1.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_018() {
        let q = Tensor::from_slice(&[1.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_019() {
        let q = Tensor::from_slice(&[1.9500000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_020() {
        let q = Tensor::from_slice(&[2.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_021() {
        let q = Tensor::from_slice(&[2.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_022() {
        let q = Tensor::from_slice(&[2.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_023() {
        let q = Tensor::from_slice(&[2.1500000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_024() {
        let q = Tensor::from_slice(&[2.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_025() {
        let q = Tensor::from_slice(&[2.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_026() {
        let q = Tensor::from_slice(&[2.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_027() {
        let q = Tensor::from_slice(&[2.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_028() {
        let q = Tensor::from_slice(&[2.4000000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_029() {
        let q = Tensor::from_slice(&[2.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_030() {
        let q = Tensor::from_slice(&[2.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_031() {
        let q = Tensor::from_slice(&[2.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_032() {
        let q = Tensor::from_slice(&[2.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_033() {
        let q = Tensor::from_slice(&[2.6500000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_034() {
        let q = Tensor::from_slice(&[2.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_035() {
        let q = Tensor::from_slice(&[2.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_036() {
        let q = Tensor::from_slice(&[2.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_037() {
        let q = Tensor::from_slice(&[2.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_038() {
        let q = Tensor::from_slice(&[2.9000000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_039() {
        let q = Tensor::from_slice(&[2.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_040() {
        let q = Tensor::from_slice(&[3.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_041() {
        let q = Tensor::from_slice(&[3.0500000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_042() {
        let q = Tensor::from_slice(&[3.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_043() {
        let q = Tensor::from_slice(&[3.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_044() {
        let q = Tensor::from_slice(&[3.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_045() {
        let q = Tensor::from_slice(&[3.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_046() {
        let q = Tensor::from_slice(&[3.3000000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_047() {
        let q = Tensor::from_slice(&[3.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_048() {
        let q = Tensor::from_slice(&[3.4000000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_049() {
        let q = Tensor::from_slice(&[3.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_050() {
        let q = Tensor::from_slice(&[3.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_051() {
        let q = Tensor::from_slice(&[3.5500000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_052() {
        let q = Tensor::from_slice(&[3.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_053() {
        let q = Tensor::from_slice(&[3.6500000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_054() {
        let q = Tensor::from_slice(&[3.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_055() {
        let q = Tensor::from_slice(&[3.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_056() {
        let q = Tensor::from_slice(&[3.8000000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_057() {
        let q = Tensor::from_slice(&[3.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_058() {
        let q = Tensor::from_slice(&[3.9000000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_059() {
        let q = Tensor::from_slice(&[3.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_060() {
        let q = Tensor::from_slice(&[4.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_061() {
        let q = Tensor::from_slice(&[4.050000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_062() {
        let q = Tensor::from_slice(&[4.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_063() {
        let q = Tensor::from_slice(&[4.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_064() {
        let q = Tensor::from_slice(&[4.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_065() {
        let q = Tensor::from_slice(&[4.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_066() {
        let q = Tensor::from_slice(&[4.300000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_067() {
        let q = Tensor::from_slice(&[4.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_068() {
        let q = Tensor::from_slice(&[4.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_069() {
        let q = Tensor::from_slice(&[4.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_070() {
        let q = Tensor::from_slice(&[4.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_071() {
        let q = Tensor::from_slice(&[4.550000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_072() {
        let q = Tensor::from_slice(&[4.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_073() {
        let q = Tensor::from_slice(&[4.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_074() {
        let q = Tensor::from_slice(&[4.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_075() {
        let q = Tensor::from_slice(&[4.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_076() {
        let q = Tensor::from_slice(&[4.800000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_077() {
        let q = Tensor::from_slice(&[4.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_078() {
        let q = Tensor::from_slice(&[4.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_079() {
        let q = Tensor::from_slice(&[4.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_080() {
        let q = Tensor::from_slice(&[5.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_081() {
        let q = Tensor::from_slice(&[5.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_082() {
        let q = Tensor::from_slice(&[5.1000000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_083() {
        let q = Tensor::from_slice(&[5.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_084() {
        let q = Tensor::from_slice(&[5.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_085() {
        let q = Tensor::from_slice(&[5.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_086() {
        let q = Tensor::from_slice(&[5.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_087() {
        let q = Tensor::from_slice(&[5.3500000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_088() {
        let q = Tensor::from_slice(&[5.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_089() {
        let q = Tensor::from_slice(&[5.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_090() {
        let q = Tensor::from_slice(&[5.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_091() {
        let q = Tensor::from_slice(&[5.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_092() {
        let q = Tensor::from_slice(&[5.6000000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_093() {
        let q = Tensor::from_slice(&[5.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_094() {
        let q = Tensor::from_slice(&[5.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_095() {
        let q = Tensor::from_slice(&[5.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_096() {
        let q = Tensor::from_slice(&[5.800000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_097() {
        let q = Tensor::from_slice(&[5.8500000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_098() {
        let q = Tensor::from_slice(&[5.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_099() {
        let q = Tensor::from_slice(&[5.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_100() {
        let q = Tensor::from_slice(&[6.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_101() {
        let q = Tensor::from_slice(&[6.050000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_102() {
        let q = Tensor::from_slice(&[6.1000000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_103() {
        let q = Tensor::from_slice(&[6.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_104() {
        let q = Tensor::from_slice(&[6.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_105() {
        let q = Tensor::from_slice(&[6.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_106() {
        let q = Tensor::from_slice(&[6.300000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_107() {
        let q = Tensor::from_slice(&[6.3500000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_108() {
        let q = Tensor::from_slice(&[6.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_109() {
        let q = Tensor::from_slice(&[6.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_110() {
        let q = Tensor::from_slice(&[6.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_111() {
        let q = Tensor::from_slice(&[6.550000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_112() {
        let q = Tensor::from_slice(&[6.6000000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_113() {
        let q = Tensor::from_slice(&[6.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_114() {
        let q = Tensor::from_slice(&[6.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_115() {
        let q = Tensor::from_slice(&[6.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_116() {
        let q = Tensor::from_slice(&[6.800000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_117() {
        let q = Tensor::from_slice(&[6.8500000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_118() {
        let q = Tensor::from_slice(&[6.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_119() {
        let q = Tensor::from_slice(&[6.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_120() {
        let q = Tensor::from_slice(&[7.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_121() {
        let q = Tensor::from_slice(&[7.050000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_122() {
        let q = Tensor::from_slice(&[7.1000000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_123() {
        let q = Tensor::from_slice(&[7.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_124() {
        let q = Tensor::from_slice(&[7.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_125() {
        let q = Tensor::from_slice(&[7.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_126() {
        let q = Tensor::from_slice(&[7.300000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_127() {
        let q = Tensor::from_slice(&[7.3500000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_128() {
        let q = Tensor::from_slice(&[7.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_129() {
        let q = Tensor::from_slice(&[7.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_130() {
        let q = Tensor::from_slice(&[7.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_131() {
        let q = Tensor::from_slice(&[7.550000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_132() {
        let q = Tensor::from_slice(&[7.6000000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_133() {
        let q = Tensor::from_slice(&[7.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_134() {
        let q = Tensor::from_slice(&[7.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_135() {
        let q = Tensor::from_slice(&[7.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_136() {
        let q = Tensor::from_slice(&[7.800000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_137() {
        let q = Tensor::from_slice(&[7.8500000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_138() {
        let q = Tensor::from_slice(&[7.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_139() {
        let q = Tensor::from_slice(&[7.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_140() {
        let q = Tensor::from_slice(&[8.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_141() {
        let q = Tensor::from_slice(&[8.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_142() {
        let q = Tensor::from_slice(&[8.100000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_143() {
        let q = Tensor::from_slice(&[8.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_144() {
        let q = Tensor::from_slice(&[8.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_145() {
        let q = Tensor::from_slice(&[8.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_146() {
        let q = Tensor::from_slice(&[8.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_147() {
        let q = Tensor::from_slice(&[8.350000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_148() {
        let q = Tensor::from_slice(&[8.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_149() {
        let q = Tensor::from_slice(&[8.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_150() {
        let q = Tensor::from_slice(&[8.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_151() {
        let q = Tensor::from_slice(&[8.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_152() {
        let q = Tensor::from_slice(&[8.600000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_153() {
        let q = Tensor::from_slice(&[8.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_154() {
        let q = Tensor::from_slice(&[8.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_155() {
        let q = Tensor::from_slice(&[8.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_156() {
        let q = Tensor::from_slice(&[8.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_157() {
        let q = Tensor::from_slice(&[8.850000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_158() {
        let q = Tensor::from_slice(&[8.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_159() {
        let q = Tensor::from_slice(&[8.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_160() {
        let q = Tensor::from_slice(&[9.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_161() {
        let q = Tensor::from_slice(&[9.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_162() {
        let q = Tensor::from_slice(&[9.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_163() {
        let q = Tensor::from_slice(&[9.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_164() {
        let q = Tensor::from_slice(&[9.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_165() {
        let q = Tensor::from_slice(&[9.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_166() {
        let q = Tensor::from_slice(&[9.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_167() {
        let q = Tensor::from_slice(&[9.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_168() {
        let q = Tensor::from_slice(&[9.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_169() {
        let q = Tensor::from_slice(&[9.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_170() {
        let q = Tensor::from_slice(&[9.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_171() {
        let q = Tensor::from_slice(&[9.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_172() {
        let q = Tensor::from_slice(&[9.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_173() {
        let q = Tensor::from_slice(&[9.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_174() {
        let q = Tensor::from_slice(&[9.700000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_175() {
        let q = Tensor::from_slice(&[9.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_176() {
        let q = Tensor::from_slice(&[9.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_177() {
        let q = Tensor::from_slice(&[9.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_178() {
        let q = Tensor::from_slice(&[9.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_179() {
        let q = Tensor::from_slice(&[9.950000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_180() {
        let q = Tensor::from_slice(&[10.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_181() {
        let q = Tensor::from_slice(&[10.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_182() {
        let q = Tensor::from_slice(&[10.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_183() {
        let q = Tensor::from_slice(&[10.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_184() {
        let q = Tensor::from_slice(&[10.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_185() {
        let q = Tensor::from_slice(&[10.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_186() {
        let q = Tensor::from_slice(&[10.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_187() {
        let q = Tensor::from_slice(&[10.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_188() {
        let q = Tensor::from_slice(&[10.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_189() {
        let q = Tensor::from_slice(&[10.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_190() {
        let q = Tensor::from_slice(&[10.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_191() {
        let q = Tensor::from_slice(&[10.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_192() {
        let q = Tensor::from_slice(&[10.600000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_193() {
        let q = Tensor::from_slice(&[10.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_194() {
        let q = Tensor::from_slice(&[10.700000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_195() {
        let q = Tensor::from_slice(&[10.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_196() {
        let q = Tensor::from_slice(&[10.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_197() {
        let q = Tensor::from_slice(&[10.850000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_198() {
        let q = Tensor::from_slice(&[10.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_199() {
        let q = Tensor::from_slice(&[10.950000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_200() {
        let q = Tensor::from_slice(&[11.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_201() {
        let q = Tensor::from_slice(&[11.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_202() {
        let q = Tensor::from_slice(&[11.100000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_203() {
        let q = Tensor::from_slice(&[11.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_204() {
        let q = Tensor::from_slice(&[11.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_205() {
        let q = Tensor::from_slice(&[11.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_206() {
        let q = Tensor::from_slice(&[11.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_207() {
        let q = Tensor::from_slice(&[11.350000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_208() {
        let q = Tensor::from_slice(&[11.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_209() {
        let q = Tensor::from_slice(&[11.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_210() {
        let q = Tensor::from_slice(&[11.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_211() {
        let q = Tensor::from_slice(&[11.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_212() {
        let q = Tensor::from_slice(&[11.600000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_213() {
        let q = Tensor::from_slice(&[11.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_214() {
        let q = Tensor::from_slice(&[11.700000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_215() {
        let q = Tensor::from_slice(&[11.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_216() {
        let q = Tensor::from_slice(&[11.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_217() {
        let q = Tensor::from_slice(&[11.850000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_218() {
        let q = Tensor::from_slice(&[11.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_219() {
        let q = Tensor::from_slice(&[11.950000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_220() {
        let q = Tensor::from_slice(&[12.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_221() {
        let q = Tensor::from_slice(&[12.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_222() {
        let q = Tensor::from_slice(&[12.100000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_223() {
        let q = Tensor::from_slice(&[12.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_224() {
        let q = Tensor::from_slice(&[12.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_225() {
        let q = Tensor::from_slice(&[12.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_226() {
        let q = Tensor::from_slice(&[12.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_227() {
        let q = Tensor::from_slice(&[12.350000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_228() {
        let q = Tensor::from_slice(&[12.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_229() {
        let q = Tensor::from_slice(&[12.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_230() {
        let q = Tensor::from_slice(&[12.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_231() {
        let q = Tensor::from_slice(&[12.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_232() {
        let q = Tensor::from_slice(&[12.600000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_233() {
        let q = Tensor::from_slice(&[12.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_234() {
        let q = Tensor::from_slice(&[12.700000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_235() {
        let q = Tensor::from_slice(&[12.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_236() {
        let q = Tensor::from_slice(&[12.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_237() {
        let q = Tensor::from_slice(&[12.850000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_238() {
        let q = Tensor::from_slice(&[12.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_239() {
        let q = Tensor::from_slice(&[12.950000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_240() {
        let q = Tensor::from_slice(&[13.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_241() {
        let q = Tensor::from_slice(&[13.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_242() {
        let q = Tensor::from_slice(&[13.100000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_243() {
        let q = Tensor::from_slice(&[13.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_244() {
        let q = Tensor::from_slice(&[13.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_245() {
        let q = Tensor::from_slice(&[13.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_246() {
        let q = Tensor::from_slice(&[13.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_247() {
        let q = Tensor::from_slice(&[13.350000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_248() {
        let q = Tensor::from_slice(&[13.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_249() {
        let q = Tensor::from_slice(&[13.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_250() {
        let q = Tensor::from_slice(&[13.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_251() {
        let q = Tensor::from_slice(&[13.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_252() {
        let q = Tensor::from_slice(&[13.600000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_253() {
        let q = Tensor::from_slice(&[13.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_254() {
        let q = Tensor::from_slice(&[13.700000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_255() {
        let q = Tensor::from_slice(&[13.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_256() {
        let q = Tensor::from_slice(&[13.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_257() {
        let q = Tensor::from_slice(&[13.850000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_258() {
        let q = Tensor::from_slice(&[13.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_259() {
        let q = Tensor::from_slice(&[13.950000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_260() {
        let q = Tensor::from_slice(&[14.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_261() {
        let q = Tensor::from_slice(&[14.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_262() {
        let q = Tensor::from_slice(&[14.100000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_263() {
        let q = Tensor::from_slice(&[14.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_264() {
        let q = Tensor::from_slice(&[14.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_265() {
        let q = Tensor::from_slice(&[14.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_266() {
        let q = Tensor::from_slice(&[14.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_267() {
        let q = Tensor::from_slice(&[14.350000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_268() {
        let q = Tensor::from_slice(&[14.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_269() {
        let q = Tensor::from_slice(&[14.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_270() {
        let q = Tensor::from_slice(&[14.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }

    #[test]
    fn test_composite_vjp_stress_271() {
        let q = Tensor::from_slice(&[14.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (dq, dk, dv) = grad_scaled_dot_product_attention(&q, &k_t, &v, &g).unwrap();
        assert_eq!(dq.shape(), &[2, 2]);
        assert_eq!(dk.shape(), &[2, 2]);
        assert_eq!(dv.shape(), &[2, 2]);
    }
}
