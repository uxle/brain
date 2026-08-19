//! # Multi-Head Attention (MHA) Architecture
//!
//! Multi-Head Attention layer with fused linear projections, multi-head splitting and merging, and causal/padding mask support.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::attention::scaled::scaled_dot_product_attention;
use crate::attention::{Attention, AttentionKind};
use crate::core::{AttentionMask, LinearParams, TransformerError, TransformerResult};
use brain_core::Tensor;

/// Configuration for Multi-Head Attention.
#[derive(Debug, Clone, PartialEq)]
pub struct MhaConfig {
    /// Hidden dimension of input/output representation.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each individual attention head.
    pub head_dim: usize,
    /// Attention dropout rate.
    pub dropout: f32,
    /// Include bias in linear projections.
    pub bias: bool,
    /// Enforce causal triangular masking.
    pub is_causal: bool,
}

impl Default for MhaConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            num_heads: 12,
            head_dim: 64,
            dropout: 0.0,
            bias: false,
            is_causal: false,
        }
    }
}

/// Production Multi-Head Attention (MHA) Layer.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Query projection parameters.
    pub q_proj: LinearParams,
    /// Key projection parameters.
    pub k_proj: LinearParams,
    /// Value projection parameters.
    pub v_proj: LinearParams,
    /// Output projection parameters.
    pub out_proj: LinearParams,
    /// Configuration options.
    pub config: MhaConfig,
}

impl MultiHeadAttention {
    /// Creates a new `MultiHeadAttention` layer with Xavier initialized projection weights.
    pub fn new(config: MhaConfig, seed: u64) -> Self {
        let q_proj = LinearParams::new(config.hidden_dim, config.hidden_dim, config.bias, seed);
        let k_proj = LinearParams::new(config.hidden_dim, config.hidden_dim, config.bias, seed.wrapping_add(100));
        let v_proj = LinearParams::new(config.hidden_dim, config.hidden_dim, config.bias, seed.wrapping_add(200));
        let out_proj = LinearParams::new(config.hidden_dim, config.hidden_dim, config.bias, seed.wrapping_add(300));

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            config,
        }
    }

    /// Splits 3D projected tensor `[batch_size, seq_len, hidden_dim]` into 4D `[batch_size, num_heads, seq_len, head_dim]`.
    pub fn split_heads(&self, tensor: &Tensor) -> TransformerResult<Tensor> {
        let shape = tensor.shape();
        if shape.len() != 3 {
            return Err(TransformerError::DimensionMismatch {
                expected: 3,
                found: shape.len(),
            });
        }
        let batch_size = shape[0];
        let seq_len = shape[1];
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        let in_data = tensor.data();
        let mut out_data = vec![0.0f64; batch_size * num_heads * seq_len * head_dim];

        for b in 0..batch_size {
            for s in 0..seq_len {
                let in_offset = b * seq_len * (num_heads * head_dim) + s * (num_heads * head_dim);
                for h in 0..num_heads {
                    let out_offset = b * (num_heads * seq_len * head_dim)
                        + h * (seq_len * head_dim)
                        + s * head_dim;
                    let h_in_offset = in_offset + h * head_dim;
                    out_data[out_offset..out_offset + head_dim]
                        .copy_from_slice(&in_data[h_in_offset..h_in_offset + head_dim]);
                }
            }
        }

        Ok(Tensor::from_vec(out_data, vec![batch_size, num_heads, seq_len, head_dim]))
    }

    /// Merges 4D attention outputs `[batch_size, num_heads, seq_len, head_dim]` back into 3D `[batch_size, seq_len, hidden_dim]`.
    pub fn merge_heads(&self, tensor: &Tensor) -> TransformerResult<Tensor> {
        let shape = tensor.shape();
        if shape.len() != 4 {
            return Err(TransformerError::DimensionMismatch {
                expected: 4,
                found: shape.len(),
            });
        }
        let batch_size = shape[0];
        let num_heads = shape[1];
        let seq_len = shape[2];
        let head_dim = shape[3];
        let hidden_dim = num_heads * head_dim;

        let in_data = tensor.data();
        let mut out_data = vec![0.0f64; batch_size * seq_len * hidden_dim];

        for b in 0..batch_size {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    let in_offset = b * (num_heads * seq_len * head_dim)
                        + h * (seq_len * head_dim)
                        + s * head_dim;
                    let out_offset = b * (seq_len * hidden_dim) + s * hidden_dim + h * head_dim;
                    out_data[out_offset..out_offset + head_dim]
                        .copy_from_slice(&in_data[in_offset..in_offset + head_dim]);
                }
            }
        }

        Ok(Tensor::from_vec(out_data, vec![batch_size, seq_len, hidden_dim]))
    }

    /// Executes complete Multi-Head Attention pass.
    pub fn forward_mha(
        &self,
        hidden_states: &Tensor,
        key_value_states: Option<&Tensor>,
        mask: &AttentionMask,
    ) -> TransformerResult<Tensor> {
        let kv_source = key_value_states.unwrap_or(hidden_states);

        let q_proj = self.q_proj.forward(hidden_states)?;
        let k_proj = self.k_proj.forward(kv_source)?;
        let v_proj = self.v_proj.forward(kv_source)?;

        let q_heads = self.split_heads(&q_proj)?;
        let k_heads = self.split_heads(&k_proj)?;
        let v_heads = self.split_heads(&v_proj)?;

        let (attn_out, _) = scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, mask, None)?;
        let merged = self.merge_heads(&attn_out)?;
        self.out_proj.forward(&merged)
    }
}

impl Attention for MultiHeadAttention {
    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: &AttentionMask,
    ) -> TransformerResult<Tensor> {
        let q_proj = self.q_proj.forward(query)?;
        let k_proj = self.k_proj.forward(key)?;
        let v_proj = self.v_proj.forward(value)?;

        let q_heads = self.split_heads(&q_proj)?;
        let k_heads = self.split_heads(&k_proj)?;
        let v_heads = self.split_heads(&v_proj)?;

        let (attn_out, _) = scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, mask, None)?;
        let merged = self.merge_heads(&attn_out)?;
        self.out_proj.forward(&merged)
    }

    fn kind(&self) -> AttentionKind {
        AttentionKind::MultiHead
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision, clippy::float_cmp, clippy::len_zero, clippy::all)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::attention::*;
    use crate::attention::scaled::*;
    use crate::attention::multi_head::*;
    use crate::attention::relative::*;
    use crate::attention::flash_lite::*;
    use crate::attention::multi_query::*;
    use crate::attention::xformers_lite::*;
    use crate::position::*;
    use crate::position::rope::*;
    use crate::position::alibi::*;
    use crate::position::learned::*;
    use crate::embedding_layers::*;
    use crate::ffn::*;
    use crate::encoder::*;
    use crate::encoder::block::*;
    use crate::encoder::layer::*;
    use crate::decoder::*;
    use crate::decoder::layer::*;
    use crate::decoder::cross::*;
    use crate::head::*;
    use crate::kv_cache::*;
    use crate::generate::*;
    use crate::models::*;
    use crate::models::bert_lite::*;
    use crate::models::gpt_lite::*;
    use crate::models::t5_lite::*;
    use crate::models::llama_lite::*;
    use crate::builder::*;
    use brain_core::Tensor;

    #[test]
    fn test_multi_head_attention_1() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }
}
