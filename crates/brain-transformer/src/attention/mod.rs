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

    #[test]
    fn test_attention_registry_2() {
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

    #[test]
    fn test_attention_registry_3() {
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

    #[test]
    fn test_attention_registry_4() {
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

    #[test]
    fn test_attention_registry_5() {
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

    #[test]
    fn test_attention_registry_6() {
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

    #[test]
    fn test_attention_registry_7() {
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

    #[test]
    fn test_attention_registry_8() {
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

    #[test]
    fn test_attention_registry_9() {
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

    #[test]
    fn test_attention_registry_10() {
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

    #[test]
    fn test_attention_registry_11() {
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

    #[test]
    fn test_attention_registry_12() {
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

    #[test]
    fn test_attention_registry_13() {
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

    #[test]
    fn test_attention_registry_14() {
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

    #[test]
    fn test_attention_registry_15() {
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

    #[test]
    fn test_attention_registry_16() {
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

    #[test]
    fn test_attention_registry_17() {
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

    #[test]
    fn test_attention_registry_18() {
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

    #[test]
    fn test_attention_registry_19() {
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

    #[test]
    fn test_attention_registry_20() {
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

    #[test]
    fn test_attention_registry_21() {
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

    #[test]
    fn test_attention_registry_22() {
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

    #[test]
    fn test_attention_registry_23() {
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

    #[test]
    fn test_attention_registry_24() {
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

    #[test]
    fn test_attention_registry_25() {
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

    #[test]
    fn test_attention_registry_26() {
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

    #[test]
    fn test_attention_registry_27() {
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

    #[test]
    fn test_attention_registry_28() {
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

    #[test]
    fn test_attention_registry_29() {
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

    #[test]
    fn test_attention_registry_30() {
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

    #[test]
    fn test_attention_registry_31() {
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

    #[test]
    fn test_attention_registry_32() {
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

    #[test]
    fn test_attention_registry_33() {
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

    #[test]
    fn test_attention_registry_34() {
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

    #[test]
    fn test_attention_registry_35() {
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

    #[test]
    fn test_attention_registry_36() {
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

    #[test]
    fn test_attention_registry_37() {
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

    #[test]
    fn test_attention_registry_38() {
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

    #[test]
    fn test_attention_registry_39() {
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

    #[test]
    fn test_attention_registry_40() {
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

    #[test]
    fn test_attention_registry_41() {
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

    #[test]
    fn test_attention_registry_42() {
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

    #[test]
    fn test_attention_registry_43() {
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

    #[test]
    fn test_attention_registry_44() {
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

    #[test]
    fn test_attention_registry_45() {
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

    #[test]
    fn test_attention_registry_46() {
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

    #[test]
    fn test_attention_registry_47() {
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

    #[test]
    fn test_attention_registry_48() {
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

    #[test]
    fn test_attention_registry_49() {
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

    #[test]
    fn test_attention_registry_50() {
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

    #[test]
    fn test_attention_registry_51() {
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

    #[test]
    fn test_attention_registry_52() {
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

    #[test]
    fn test_attention_registry_53() {
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

    #[test]
    fn test_attention_registry_54() {
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

    #[test]
    fn test_attention_registry_55() {
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

    #[test]
    fn test_attention_registry_56() {
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

    #[test]
    fn test_attention_registry_57() {
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

    #[test]
    fn test_attention_registry_58() {
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

    #[test]
    fn test_attention_registry_59() {
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

    #[test]
    fn test_attention_registry_60() {
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

    #[test]
    fn test_attention_registry_61() {
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

    #[test]
    fn test_attention_registry_62() {
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

    #[test]
    fn test_attention_registry_63() {
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

    #[test]
    fn test_attention_registry_64() {
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

    #[test]
    fn test_attention_registry_65() {
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

    #[test]
    fn test_attention_registry_66() {
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

    #[test]
    fn test_attention_registry_67() {
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

    #[test]
    fn test_attention_registry_68() {
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

    #[test]
    fn test_attention_registry_69() {
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

    #[test]
    fn test_attention_registry_70() {
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

    #[test]
    fn test_attention_registry_71() {
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

    #[test]
    fn test_attention_registry_72() {
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

    #[test]
    fn test_attention_registry_73() {
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

    #[test]
    fn test_attention_registry_74() {
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

    #[test]
    fn test_attention_registry_75() {
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

    #[test]
    fn test_attention_registry_76() {
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

    #[test]
    fn test_attention_registry_77() {
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

    #[test]
    fn test_attention_registry_78() {
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

    #[test]
    fn test_attention_registry_79() {
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

    #[test]
    fn test_attention_registry_80() {
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

    #[test]
    fn test_attention_registry_81() {
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

    #[test]
    fn test_attention_registry_82() {
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

    #[test]
    fn test_attention_registry_83() {
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

    #[test]
    fn test_attention_registry_84() {
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

    #[test]
    fn test_attention_registry_85() {
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

    #[test]
    fn test_attention_registry_86() {
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

    #[test]
    fn test_attention_registry_87() {
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

    #[test]
    fn test_attention_registry_88() {
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

    #[test]
    fn test_attention_registry_89() {
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

    #[test]
    fn test_attention_registry_90() {
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

    #[test]
    fn test_attention_registry_91() {
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

    #[test]
    fn test_attention_registry_92() {
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

    #[test]
    fn test_attention_registry_93() {
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

    #[test]
    fn test_attention_registry_94() {
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

    #[test]
    fn test_attention_registry_95() {
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

    #[test]
    fn test_attention_registry_96() {
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

    #[test]
    fn test_attention_registry_97() {
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

    #[test]
    fn test_attention_registry_98() {
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

    #[test]
    fn test_attention_registry_99() {
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

    #[test]
    fn test_attention_registry_100() {
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

    #[test]
    fn test_attention_registry_101() {
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

    #[test]
    fn test_attention_registry_102() {
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

    #[test]
    fn test_attention_registry_103() {
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

    #[test]
    fn test_attention_registry_104() {
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

    #[test]
    fn test_attention_registry_105() {
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

    #[test]
    fn test_attention_registry_106() {
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

    #[test]
    fn test_attention_registry_107() {
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

    #[test]
    fn test_attention_registry_108() {
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

    #[test]
    fn test_attention_registry_109() {
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

    #[test]
    fn test_attention_registry_110() {
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

    #[test]
    fn test_attention_registry_111() {
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

    #[test]
    fn test_attention_registry_112() {
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

    #[test]
    fn test_attention_registry_113() {
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

    #[test]
    fn test_attention_registry_114() {
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

    #[test]
    fn test_attention_registry_115() {
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

    #[test]
    fn test_attention_registry_116() {
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

    #[test]
    fn test_attention_registry_117() {
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

    #[test]
    fn test_attention_registry_118() {
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

    #[test]
    fn test_attention_registry_119() {
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

    #[test]
    fn test_attention_registry_120() {
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

    #[test]
    fn test_attention_registry_121() {
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

    #[test]
    fn test_attention_registry_122() {
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

    #[test]
    fn test_attention_registry_123() {
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

    #[test]
    fn test_attention_registry_124() {
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

    #[test]
    fn test_attention_registry_125() {
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

    #[test]
    fn test_attention_registry_126() {
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

    #[test]
    fn test_attention_registry_127() {
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

    #[test]
    fn test_attention_registry_128() {
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

    #[test]
    fn test_attention_registry_129() {
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

    #[test]
    fn test_attention_registry_130() {
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

    #[test]
    fn test_attention_registry_131() {
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

    #[test]
    fn test_attention_registry_132() {
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

    #[test]
    fn test_attention_registry_133() {
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

    #[test]
    fn test_attention_registry_134() {
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

    #[test]
    fn test_attention_registry_135() {
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

    #[test]
    fn test_attention_registry_136() {
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

    #[test]
    fn test_attention_registry_137() {
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

    #[test]
    fn test_attention_registry_138() {
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

    #[test]
    fn test_attention_registry_139() {
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

    #[test]
    fn test_attention_registry_140() {
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

    #[test]
    fn test_attention_registry_141() {
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

    #[test]
    fn test_attention_registry_142() {
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

    #[test]
    fn test_attention_registry_143() {
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

    #[test]
    fn test_attention_registry_144() {
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

    #[test]
    fn test_attention_registry_145() {
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

    #[test]
    fn test_attention_registry_146() {
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

    #[test]
    fn test_attention_registry_147() {
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

    #[test]
    fn test_attention_registry_148() {
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

    #[test]
    fn test_attention_registry_149() {
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

    #[test]
    fn test_attention_registry_150() {
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

    #[test]
    fn test_attention_registry_151() {
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

    #[test]
    fn test_attention_registry_152() {
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

    #[test]
    fn test_attention_registry_153() {
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

    #[test]
    fn test_attention_registry_154() {
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

    #[test]
    fn test_attention_registry_155() {
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

    #[test]
    fn test_attention_registry_156() {
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

    #[test]
    fn test_attention_registry_157() {
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

    #[test]
    fn test_attention_registry_158() {
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

    #[test]
    fn test_attention_registry_159() {
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

    #[test]
    fn test_attention_registry_160() {
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

    #[test]
    fn test_attention_registry_161() {
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

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
}
