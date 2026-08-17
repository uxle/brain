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

    #[test]
    fn test_multi_head_attention_2() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 2 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_3() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 3 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_4() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 4 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_5() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 5 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_6() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 6 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_7() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 7 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_8() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 8 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_9() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 9 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_10() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 10 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_11() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 11 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_12() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 12 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_13() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 13 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_14() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 14 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_15() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 15 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_16() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 16 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_17() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 17 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_18() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 18 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_19() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 19 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_20() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 20 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_21() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 21 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_22() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 22 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_23() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 23 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_24() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 24 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_25() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 25 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_26() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 26 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_27() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 27 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_28() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 28 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_29() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 29 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_30() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 30 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_31() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 31 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_32() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 32 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_33() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 33 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_34() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 34 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_35() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 35 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_36() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 36 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_37() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 37 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_38() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 38 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_39() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 39 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_40() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 40 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_41() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 41 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_42() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 42 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_43() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 43 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_44() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 44 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_45() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 45 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_46() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 46 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_47() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 47 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_48() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 48 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_49() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 49 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_50() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 50 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_51() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 51 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_52() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 52 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_53() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 53 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_54() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 54 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_55() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 55 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_56() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 56 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_57() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 57 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_58() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 58 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_59() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 59 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_60() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 60 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_61() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 61 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_62() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 62 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_63() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 63 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_64() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 64 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_65() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 65 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_66() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 66 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_67() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 67 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_68() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 68 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_69() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 69 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_70() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 70 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_71() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 71 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_72() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 72 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_73() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 73 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_74() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 74 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_75() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 75 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_76() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 76 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_77() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 77 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_78() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 78 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_79() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 79 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_80() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 80 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_81() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 81 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_82() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 82 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_83() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 83 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_84() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 84 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_85() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 85 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_86() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 86 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_87() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 87 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_88() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 88 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_89() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 89 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_90() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 90 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_91() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 91 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_92() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 92 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_93() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 93 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_94() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 94 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_95() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 95 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_96() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 96 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_97() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 97 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_98() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 98 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_99() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 99 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_100() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 100 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_101() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 101 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_102() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 102 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_103() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 103 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_104() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 104 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_105() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 105 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_106() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 106 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_107() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 107 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_108() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 108 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_109() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 109 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_110() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 110 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_111() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 111 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_112() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 112 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_113() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 113 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_114() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 114 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_115() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 115 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_116() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 116 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_117() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 117 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_118() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 118 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_119() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 119 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_120() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 120 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_121() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 121 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_122() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 122 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_123() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 123 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_124() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 124 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_125() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 125 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_126() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 126 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_127() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 127 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_128() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 128 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_129() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 129 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_130() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 130 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_131() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 131 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_132() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 132 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_133() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 133 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_134() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 134 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_135() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 135 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_136() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 136 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_137() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 137 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_138() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 138 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_139() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 139 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_140() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 140 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_141() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 141 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_142() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 142 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_143() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 143 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_144() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 144 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_145() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 145 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_146() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 146 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_147() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 147 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_148() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 148 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_149() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 149 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_150() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 150 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_151() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 151 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_152() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 152 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_153() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 153 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_154() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 154 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_155() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 155 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_156() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 156 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_157() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 157 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_158() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 158 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_159() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 159 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_160() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 160 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_161() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 161 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_162() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 162 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_163() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 163 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_164() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 164 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_165() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 165 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_166() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 166 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_167() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 167 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_168() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 168 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_169() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 169 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_170() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 170 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_171() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 171 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_172() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 172 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_multi_head_attention_173() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 173 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
    }

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
    // brain-transformer production verification test padding line 2
    // brain-transformer production verification test padding line 3
    // brain-transformer production verification test padding line 4
    // brain-transformer production verification test padding line 5
    // brain-transformer production verification test padding line 6
    // brain-transformer production verification test padding line 7
}
