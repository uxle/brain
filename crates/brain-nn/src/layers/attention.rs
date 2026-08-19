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
    mask: Option<&Tensor>,
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
    let mask_data = mask.map(|m| m.to_vec());

    let mut out_data = vec![0.0f64; batch * seq_q * d_v];

    for b in 0..batch {
        for sq in 0..seq_q {
            let mut scores = vec![0.0f64; seq_k];
            for sk in 0..seq_k {
                let mut dot = 0.0f64;
                for d in 0..d_k {
                    dot += q_data[b * seq_q * d_k + sq * d_k + d] * k_data[b * seq_k * d_k + sk * d_k + d];
                }
                let mut s = dot * scale;
                if let Some(ref m) = mask_data {
                    let m_idx = if m.len() == seq_q * seq_k {
                        sq * seq_k + sk
                    } else if m.len() == batch * seq_q * seq_k {
                        (b * seq_q + sq) * seq_k + sk
                    } else if m.len() == seq_k {
                        sk
                    } else {
                        0
                    };
                    if m_idx < m.len() {
                        if m[m_idx] == 0.0 || m[m_idx] <= -1e4 {
                            s += -1e9;
                        } else {
                            s += m[m_idx];
                        }
                    }
                }
                scores[sk] = s;
            }

            // Softmax over seq_k
            let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = scores.iter().map(|&s| if s.is_finite() { (s - max_s).exp() } else { 0.0 }).sum();
            let attn_weights: Vec<f64> = scores.iter().map(|&s| {
                if s.is_finite() && exp_sum > 0.0 {
                    (s - max_s).exp() / exp_sum
                } else {
                    0.0
                }
            }).collect();

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
}
