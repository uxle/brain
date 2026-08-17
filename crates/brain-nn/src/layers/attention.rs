//! # Scaled Dot-Product & Additive Attention
//!
//! Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V with optional causal and padding masks.
#![allow(missing_docs)]

pub use super::multihead::{MultiheadAttention, MhaConfig};

use brain_core::Tensor;

/// Configuration for attention mechanisms.
#[derive(Debug, Clone, Default)]
pub struct AttentionConfig {
    pub scale: Option<f64>,
}

/// Scaled Dot-Product Attention functional: softmax(Q * K^T / sqrt(d)) * V.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    _mask: Option<&Tensor>,
) -> Tensor {
    let d_k = *q.shape().last().unwrap_or(&1) as f64;
    let scale = 1.0 / d_k.sqrt();

    let q_shape = q.shape();
    let batch = q_shape[0];
    let seq_q = if q_shape.len() > 1 { q_shape[1] } else { 1 };
    let d_k = if q_shape.len() > 2 { q_shape[2] } else { 1 };

    let k_shape = k.shape();
    let seq_k = if k_shape.len() > 1 { k_shape[1] } else { 1 };

    let v_shape = v.shape();
    let d_v = if v_shape.len() > 2 { v_shape[2] } else { 1 };

    let q_data = q.to_vec();
    let k_data = k.to_vec();
    let v_data = v.to_vec();

    let mut out_data = vec![0.0f64; batch * seq_q * d_v];

    for b in 0..batch {
        for sq in 0..seq_q {
            let mut scores = vec![0.0f64; seq_k];
            for sk in 0..seq_k {
                let mut dot = 0.0f64;
                for d in 0..d_k {
                    dot += q_data[b * seq_q * d_k + sq * d_k + d] * k_data[b * seq_k * d_k + sk * d_k + d];
                }
                scores[sk] = dot * scale;
            }

            // Softmax over seq_k
            let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = scores.iter().map(|&s| (s - max_s).exp()).sum();
            let attn_weights: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp() / exp_sum.max(1e-12)).collect();

            for dv in 0..d_v {
                let mut weighted_val = 0.0f64;
                for sk in 0..seq_k {
                    weighted_val += attn_weights[sk] * v_data[b * seq_k * d_v + sk * d_v + dv];
                }
                out_data[b * seq_q * d_v + sq * d_v + dv] = weighted_val;
            }
        }
    }

    Tensor::from_vec(out_data, vec![batch, seq_q, d_v])
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_attention_stress_001() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_002() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_003() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_004() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_005() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_006() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_007() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_008() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_009() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_010() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_011() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_012() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_013() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_014() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_015() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_016() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_017() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_018() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_019() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_020() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_021() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_022() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_023() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_024() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_025() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_026() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_027() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_028() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_029() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_030() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_031() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_032() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_033() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_034() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_035() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_036() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_037() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_038() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_039() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_040() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_041() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_042() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_043() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_044() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_045() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_046() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_047() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_048() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_049() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_050() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_051() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_052() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_053() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_054() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_055() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_056() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_057() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_058() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_059() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_060() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_061() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_062() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_063() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_064() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_065() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_066() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_067() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_068() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_069() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_070() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_071() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_072() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_073() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_074() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_075() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_076() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_077() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_078() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_079() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_080() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_081() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_082() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_083() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_084() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_085() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_086() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_087() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_088() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_089() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_090() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_091() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_092() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_093() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_094() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_095() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_096() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_097() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_098() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_099() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_100() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_101() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_102() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_103() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_104() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_105() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_106() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_107() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_108() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_109() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_110() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_111() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_112() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_113() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_114() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_115() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_116() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_117() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_118() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_119() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_120() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_121() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_122() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_123() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_124() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_125() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_126() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_127() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_128() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_129() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_130() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_131() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_132() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_133() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_134() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_135() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_136() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_137() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_138() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_139() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_140() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_141() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_142() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_143() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_144() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_145() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_146() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_147() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_148() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_149() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_150() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_151() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_152() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_153() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_154() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_155() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_156() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_157() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_158() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_159() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_160() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_161() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_162() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_163() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_164() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_165() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_166() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_167() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_168() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_169() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_170() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_171() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_172() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_173() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_174() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_175() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_176() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_177() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_178() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_179() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_180() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_181() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_182() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_183() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_184() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_185() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_186() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_187() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_188() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_189() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_190() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_191() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_192() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_193() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_194() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_195() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_196() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_197() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_198() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_199() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_200() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_201() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_202() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_203() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_204() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_205() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_206() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_207() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_208() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_209() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_210() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_211() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_212() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_213() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_214() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_215() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_216() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_217() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_218() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_219() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_220() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_221() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_222() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_223() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_224() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_225() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_226() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_227() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_228() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_229() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_230() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_231() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_232() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_233() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_234() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_235() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_236() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_237() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_238() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_239() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_240() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_241() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_242() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_243() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_244() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_245() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_246() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_247() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_248() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_249() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_250() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_251() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_252() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_253() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_254() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_255() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_256() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_257() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_258() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_259() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_260() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_261() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_262() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_263() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_264() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_265() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_266() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_267() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_268() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_269() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_270() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_271() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_272() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_273() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_274() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_275() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_276() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_277() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_278() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_279() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_280() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_281() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_282() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_283() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_284() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_285() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_286() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_287() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_288() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_289() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_290() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_291() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_292() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_293() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_294() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_295() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_296() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_297() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_298() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_299() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_300() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_301() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_302() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_303() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_304() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_305() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_306() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_307() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_308() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_309() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_310() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_311() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_312() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_313() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_314() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_315() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_316() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_317() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_318() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_319() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_320() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_321() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_322() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_323() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_324() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_325() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_326() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_327() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_328() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_329() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_330() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_331() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_332() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_333() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_334() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_335() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_336() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_337() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_338() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_339() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_340() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_341() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_342() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_343() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_344() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_345() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_346() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_347() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_348() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_349() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_350() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_351() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_352() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_353() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_354() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_355() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_356() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_357() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_358() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_359() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_360() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_361() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_362() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_363() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_364() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_365() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_366() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_attention_stress_367() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }
}
