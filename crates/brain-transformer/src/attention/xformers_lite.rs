//! # Memory-Efficient Chunked Attention (xFormers-Lite)
//!
//! Chunked attention computation processing large sequences in small memory footprints.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{AttentionMask, TransformerError, TransformerResult};
use brain_core::Tensor;

/// Configuration for memory-efficient chunked attention.
#[derive(Debug, Clone, PartialEq)]
pub struct XformersConfig {
    /// Sequence chunk size (e.g. 64, 128, 256).
    pub chunk_size: usize,
    /// Custom scale factor.
    pub scale: Option<f64>,
    /// Enforce causal masking.
    pub is_causal: bool,
}

impl Default for XformersConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64,
            scale: None,
            is_causal: false,
        }
    }
}

/// Memory-efficient chunked attention processor.
pub struct XformersAttentionLite;

impl XformersAttentionLite {
    /// Computes memory-efficient chunked attention for 4D Tensors `[batch, heads, seq, head_dim]`.
    pub fn forward(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        config: &XformersConfig,
    ) -> TransformerResult<Tensor> {
        let flash_cfg = crate::attention::flash_lite::FlashLiteConfig {
            block_m: config.chunk_size,
            block_n: config.chunk_size,
            scale: config.scale,
            is_causal: config.is_causal,
        };
        crate::attention::flash_lite::FlashAttentionLite::forward(query, key, value, &flash_cfg)
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
    fn test_xformers_lite_1() {
        let cfg = XformersConfig { chunk_size: 4, scale: None, is_causal: false };
        let q = Tensor::from_vec(vec![1.0; 1 * 2 * 8 * 8], vec![1, 2, 8, 8]);
        let k_t = Tensor::from_vec(vec![1.0; 1 * 2 * 8 * 8], vec![1, 2, 8, 8]);
        let v = Tensor::from_vec(vec![2.0; 1 * 2 * 8 * 8], vec![1, 2, 8, 8]);

        let out = XformersAttentionLite::forward(&q, &k_t, &v, &cfg).unwrap();
        assert_eq!(out.shape(), &[1, 2, 8, 8]);
    }
}
