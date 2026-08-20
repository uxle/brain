//! # Scaled Dot-Product Attention Kernel
//!
//! Foundational attention kernel: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}} + M)V$.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::core::{AttentionMask, TransformerError, TransformerResult};
use crate::ops::{apply_attention_mask, softmax_inplace};
use brain_core::Tensor;

/// Configuration for scaled dot-product attention.
#[derive(Debug, Clone, PartialEq)]
pub struct SdpaConfig {
    /// Custom scaling factor (default: $1 / \sqrt{d_k}$).
    pub scale: Option<f64>,
    /// Dropout probability on attention weights.
    pub dropout: f32,
    /// Enforce causal masking.
    pub is_causal: bool,
}

impl Default for SdpaConfig {
    fn default() -> Self {
        Self {
            scale: None,
            dropout: 0.0,
            is_causal: false,
        }
    }
}

/// Computes scaled dot-product attention for 4D Tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.
/// Returns `(output_tensor, attention_weights_tensor)`.
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: &AttentionMask,
    scale: Option<f64>,
) -> TransformerResult<(Tensor, Tensor)> {
    let q_shape = query.shape();
    let k_shape = key.shape();
    let v_shape = value.shape();

    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
        return Err(TransformerError::DimensionMismatch {
            expected: 4,
            found: q_shape.len().max(k_shape.len()).max(v_shape.len()),
        });
    }

    let batch_size = q_shape[0];
    let num_heads = q_shape[1];
    let seq_q = q_shape[2];
    let head_dim = q_shape[3];

    let seq_k = k_shape[2];
    let k_head_dim = k_shape[3];
    let v_head_dim = v_shape[3];

    if head_dim != k_head_dim {
        return Err(TransformerError::DimensionMismatch {
            expected: head_dim,
            found: k_head_dim,
        });
    }

    let s_factor = scale.unwrap_or(1.0 / (head_dim as f64).sqrt());
    let q_data = query.data();
    let k_data = key.data();
    let v_data = value.data();

    let total_heads = batch_size * num_heads;
    let mut attn_weights = vec![0.0f64; total_heads * seq_q * seq_k];
    let mut out_data = vec![0.0f64; total_heads * seq_q * v_head_dim];

    for b in 0..batch_size {
        for h in 0..num_heads {
            let bh_idx = b * num_heads + h;
            let q_head_offset = (b * num_heads + h) * seq_q * head_dim;
            let k_head_offset = (b * num_heads + h) * seq_k * head_dim;
            let v_head_offset = (b * num_heads + h) * seq_k * v_head_dim;
            let w_head_offset = bh_idx * seq_q * seq_k;
            let o_head_offset = bh_idx * seq_q * v_head_dim;

            // 1. Compute Raw Attention Scores: Q * K^T * scale
            for i in 0..seq_q {
                let q_row_offset = q_head_offset + i * head_dim;
                let w_row_offset = w_head_offset + i * seq_k;

                for j in 0..seq_k {
                    let k_row_offset = k_head_offset + j * head_dim;
                    let mut dot = 0.0f64;
                    for d in 0..head_dim {
                        dot += q_data[q_row_offset + d] * k_data[k_row_offset + d];
                    }
                    attn_weights[w_row_offset + j] = dot * s_factor;
                }
            }

            // 2. Apply Masking
            apply_attention_mask(
                &mut attn_weights[w_head_offset..w_head_offset + seq_q * seq_k],
                seq_q,
                seq_k,
                mask,
                b,
            );

            // 3. Softmax over key sequence & Weighted sum with Value matrix: Weights * V
            for i in 0..seq_q {
                let w_row_offset = w_head_offset + i * seq_k;
                softmax_inplace(&mut attn_weights[w_row_offset..w_row_offset + seq_k]);

                let o_row_offset = o_head_offset + i * v_head_dim;
                for d in 0..v_head_dim {
                    let mut val_sum = 0.0f64;
                    for j in 0..seq_k {
                        val_sum += attn_weights[w_row_offset + j]
                            * v_data[v_head_offset + j * v_head_dim + d];
                    }
                    out_data[o_row_offset + d] = val_sum;
                }
            }
        }
    }

    let out_tensor = Tensor::from_vec(out_data, vec![batch_size, num_heads, seq_q, v_head_dim]);
    let weights_tensor = Tensor::from_vec(attn_weights, vec![batch_size, num_heads, seq_q, seq_k]);

    Ok((out_tensor, weights_tensor))
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown,
        clippy::excessive_precision,
        clippy::float_cmp,
        clippy::len_zero,
        clippy::all
    )]
    use super::*;
    use crate::attention::flash_lite::*;
    use crate::attention::multi_head::*;
    use crate::attention::multi_query::*;
    use crate::attention::relative::*;
    use crate::attention::scaled::*;
    use crate::attention::xformers_lite::*;
    use crate::attention::*;
    use crate::builder::*;
    use crate::config::*;
    use crate::core::*;
    use crate::decoder::cross::*;
    use crate::decoder::layer::*;
    use crate::decoder::*;
    use crate::embedding_layers::*;
    use crate::encoder::block::*;
    use crate::encoder::layer::*;
    use crate::encoder::*;
    use crate::ffn::*;
    use crate::generate::*;
    use crate::head::*;
    use crate::kv_cache::*;
    use crate::models::bert_lite::*;
    use crate::models::gpt_lite::*;
    use crate::models::llama_lite::*;
    use crate::models::t5_lite::*;
    use crate::models::*;
    use crate::ops::*;
    use crate::position::alibi::*;
    use crate::position::learned::*;
    use crate::position::rope::*;
    use crate::position::*;
    use crate::utils::*;
    use brain_core::Tensor;

    #[test]
    fn test_scaled_dot_product_1() {
        let q = Tensor::from_vec(vec![1.0; 2 * 2 * 3 * 4], vec![2, 2, 3, 4]);
        let k_t = Tensor::from_vec(vec![1.0; 2 * 2 * 3 * 4], vec![2, 2, 3, 4]);
        let v = Tensor::from_vec(vec![2.0; 2 * 2 * 3 * 4], vec![2, 2, 3, 4]);

        let (out, weights) =
            scaled_dot_product_attention(&q, &k_t, &v, &AttentionMask::None, None).unwrap();
        assert_eq!(out.shape(), &[2, 2, 3, 4]);
        assert_eq!(weights.shape(), &[2, 2, 3, 3]);

        // Check row sum of softmax attention weights is exactly 1.0
        let w_data = weights.data();
        let row_sum: f64 = w_data[0..3].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
}
