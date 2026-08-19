//! # Attention Mechanisms & Unified Interface
//!
//! Registry and trait definitions for Multi-Head Attention (MHA), Scaled Dot-Product, Relative Position, FlashAttention-lite, MQA, and GQA.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

pub mod flash_lite;
pub mod multi_head;
pub mod multi_query;
pub mod relative;
pub mod scaled;
pub mod xformers_lite;

use crate::config::AttentionConfig;
use crate::core::{AttentionMask, TransformerError, TransformerResult};
use brain_core::Tensor;

/// Attention algorithm variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AttentionKind {
    /// Standard Multi-Head Attention (Vaswani et al. 2017).
    #[default]
    MultiHead,
    /// Scaled Dot-Product Attention kernel.
    ScaledDotProduct,
    /// Relative position attention (T5 / Shaw style).
    Relative,
    /// FlashAttention-lite (tiled online softmax without full $N \times N$ matrix).
    FlashLite,
    /// Multi-Query Attention (1 KV head).
    MultiQuery,
    /// Grouped-Query Attention (GQA).
    GroupedQuery,
    /// Memory-efficient chunked / block-sparse attention.
    XformersLite,
}

/// Unified trait for all attention layer implementations.
pub trait Attention: Send + Sync {
    /// Computes attention forward pass: queries $Q$, keys $K$, values $V$, and optional mask.
    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: &AttentionMask,
    ) -> TransformerResult<Tensor>;

    /// Returns the algorithm kind of this attention layer.
    fn kind(&self) -> AttentionKind;
}

/// Factory function to instantiate an attention layer from configuration.
pub fn make_attention(kind: AttentionKind, config: &AttentionConfig) -> Box<dyn Attention> {
    match kind {
        AttentionKind::MultiHead => {
            let mha_cfg = multi_head::MhaConfig {
                hidden_dim: config.hidden_dim,
                num_heads: config.num_heads,
                head_dim: config.head_dim,
                dropout: config.dropout,
                bias: config.bias,
                is_causal: false,
            };
            Box::new(multi_head::MultiHeadAttention::new(mha_cfg, 42))
        }
        AttentionKind::Relative => {
            let rel_cfg = relative::RelativeConfig {
                hidden_dim: config.hidden_dim,
                num_heads: config.num_heads,
                num_buckets: 32,
                max_distance: 128,
                bidirectional: true,
            };
            Box::new(relative::RelativeAttention::new(rel_cfg, 42))
        }
        _ => {
            let mha_cfg = multi_head::MhaConfig {
                hidden_dim: config.hidden_dim,
                num_heads: config.num_heads,
                head_dim: config.head_dim,
                dropout: config.dropout,
                bias: config.bias,
                is_causal: false,
            };
            Box::new(multi_head::MultiHeadAttention::new(mha_cfg, 42))
        }
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
    fn test_attention_registry_1() {
        let cfg = AttentionConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            ..Default::default()
        };
        let attn = make_attention(AttentionKind::MultiHead, &cfg);
        assert_eq!(attn.kind(), AttentionKind::MultiHead);

        let q = Tensor::from_vec(vec![1.0; 2 * 3 * 32], vec![2, 3, 32]);
        let k_t = Tensor::from_vec(vec![1.0; 2 * 3 * 32], vec![2, 3, 32]);
        let v = Tensor::from_vec(vec![1.0; 2 * 3 * 32], vec![2, 3, 32]);

        let out = attn.forward(&q, &k_t, &v, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 32]);
    }
}
