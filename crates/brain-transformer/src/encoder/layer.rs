//! # Encoder Layer Adapters & Wrappers
//!
//! Wrapper structures for encoder blocks with layer scaling, deep norm coefficient multipliers, and inspection hooks.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{AttentionMask, TransformerResult};
use crate::encoder::block::{BlockConfig, TransformerEncoderBlock};
use brain_core::Tensor;

/// Configurable wrapper around `TransformerEncoderBlock`.
#[derive(Debug, Clone)]
pub struct EncoderLayer {
    /// Inner encoder block implementation.
    pub block: TransformerEncoderBlock,
    /// Layer index within encoder stack.
    pub layer_idx: usize,
}

impl EncoderLayer {
    /// Creates a new `EncoderLayer`.
    pub fn new(config: BlockConfig, layer_idx: usize, seed: u64) -> Self {
        let block = TransformerEncoderBlock::new(config, seed);
        Self { block, layer_idx }
    }

    /// Executes encoder layer forward pass.
    pub fn forward(&self, hidden_states: &Tensor, mask: &AttentionMask) -> TransformerResult<Tensor> {
        self.block.forward(hidden_states, mask)
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
    fn test_encoder_layer_1() {
        let cfg = BlockConfig {
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            ..Default::default()
        };
        let layer = EncoderLayer::new(cfg, 0, 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 1 * 2 * 16], vec![1, 2, 16]);
        let out = layer.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[1, 2, 16]);
    }
}
