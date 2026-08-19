//! # Multi-Query (MQA) and Grouped-Query Attention (GQA)
//!
//! KV-head sharing optimizations for high-throughput inference with reduced memory bandwidth.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::attention::scaled::scaled_dot_product_attention;
use crate::attention::{Attention, AttentionKind};
use crate::core::{AttentionMask, LinearParams, TransformerError, TransformerResult};
use brain_core::Tensor;

/// Configuration for Multi-Query Attention (MQA: 1 single KV head).
#[derive(Debug, Clone, PartialEq)]
pub struct MqaConfig {
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Dimension of each query head.
    pub head_dim: usize,
    /// Dropout rate.
    pub dropout: f32,
    /// Bias flag.
    pub bias: bool,
}

/// Configuration for Grouped-Query Attention (GQA: $G$ KV heads).
#[derive(Debug, Clone, PartialEq)]
pub struct GqaConfig {
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of query heads (e.g. 32).
    pub num_query_heads: usize,
    /// Number of key/value heads (e.g. 8).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Dropout rate.
    pub dropout: f32,
    /// Bias flag.
    pub bias: bool,
}

/// Repeats / broadcasts KV heads to match the number of query heads.
pub fn repeat_kv(tensor: &Tensor, n_rep: usize) -> TransformerResult<Tensor> {
    if n_rep == 1 {
        return Ok(tensor.clone());
    }

    let shape = tensor.shape();
    if shape.len() != 4 {
        return Err(TransformerError::DimensionMismatch {
            expected: 4,
            found: shape.len(),
        });
    }

    let batch_size = shape[0];
    let num_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    let in_data = tensor.data();
    let num_q_heads = num_kv_heads * n_rep;
    let mut out_data = vec![0.0f64; batch_size * num_q_heads * seq_len * head_dim];

    for b in 0..batch_size {
        for kv_h in 0..num_kv_heads {
            let in_head_offset = (b * num_kv_heads + kv_h) * seq_len * head_dim;
            let in_slice = &in_data[in_head_offset..in_head_offset + seq_len * head_dim];

            for r in 0..n_rep {
                let q_h = kv_h * n_rep + r;
                let out_head_offset = (b * num_q_heads + q_h) * seq_len * head_dim;
                out_data[out_head_offset..out_head_offset + seq_len * head_dim].copy_from_slice(in_slice);
            }
        }
    }

    Ok(Tensor::from_vec(out_data, vec![batch_size, num_q_heads, seq_len, head_dim]))
}

/// Grouped-Query Attention (GQA) Layer.
#[derive(Debug, Clone)]
pub struct GroupedQueryAttention {
    /// Query projection parameters.
    pub q_proj: LinearParams,
    /// Key projection parameters (projects to `num_kv_heads * head_dim`).
    pub k_proj: LinearParams,
    /// Value projection parameters (projects to `num_kv_heads * head_dim`).
    pub v_proj: LinearParams,
    /// Output projection parameters.
    pub out_proj: LinearParams,
    /// Configuration options.
    pub config: GqaConfig,
}

impl GroupedQueryAttention {
    /// Creates a new `GroupedQueryAttention` layer.
    pub fn new(config: GqaConfig, seed: u64) -> Self {
        let q_dim = config.num_query_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        let q_proj = LinearParams::new(config.hidden_dim, q_dim, config.bias, seed);
        let k_proj = LinearParams::new(config.hidden_dim, kv_dim, config.bias, seed.wrapping_add(100));
        let v_proj = LinearParams::new(config.hidden_dim, kv_dim, config.bias, seed.wrapping_add(200));
        let out_proj = LinearParams::new(q_dim, config.hidden_dim, config.bias, seed.wrapping_add(300));

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            config,
        }
    }

    /// Executes GQA forward pass.
    pub fn forward_gqa(
        &self,
        hidden_states: &Tensor,
        mask: &AttentionMask,
    ) -> TransformerResult<Tensor> {
        let shape = hidden_states.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let q_4d = Tensor::from_vec(
            q.data().to_vec(),
            vec![batch_size, self.config.num_query_heads, seq_len, self.config.head_dim],
        );
        let k_4d = Tensor::from_vec(
            k.data().to_vec(),
            vec![batch_size, self.config.num_kv_heads, seq_len, self.config.head_dim],
        );
        let v_4d = Tensor::from_vec(
            v.data().to_vec(),
            vec![batch_size, self.config.num_kv_heads, seq_len, self.config.head_dim],
        );

        let n_rep = self.config.num_query_heads / self.config.num_kv_heads;
        let k_expanded = repeat_kv(&k_4d, n_rep)?;
        let v_expanded = repeat_kv(&v_4d, n_rep)?;

        let (attn_out, _) = scaled_dot_product_attention(&q_4d, &k_expanded, &v_expanded, mask, None)?;
        let merged = Tensor::from_vec(
            attn_out.data().to_vec(),
            vec![batch_size, seq_len, self.config.num_query_heads * self.config.head_dim],
        );

        self.out_proj.forward(&merged)
    }
}

impl Attention for GroupedQueryAttention {
    fn forward(
        &self,
        query: &Tensor,
        _key: &Tensor,
        _value: &Tensor,
        mask: &AttentionMask,
    ) -> TransformerResult<Tensor> {
        self.forward_gqa(query, mask)
    }

    fn kind(&self) -> AttentionKind {
        AttentionKind::GroupedQuery
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
    fn test_gqa_mqa_attention_1() {
        let cfg = GqaConfig {
            hidden_dim: 32,
            num_query_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            bias: false,
            dropout: 0.0,
        };
        let gqa = GroupedQueryAttention::new(cfg, 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 32], vec![2, 3, 32]);
        let out = gqa.forward_gqa(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 32]);

        let kv = Tensor::from_vec(vec![1.0; 2 * 2 * 3 * 8], vec![2, 2, 3, 8]);
        let repeated = repeat_kv(&kv, 2).unwrap();
        assert_eq!(repeated.shape(), &[2, 4, 3, 8]);
    }
}
