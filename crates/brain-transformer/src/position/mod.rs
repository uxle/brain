//! # Positional Encodings & Spatial Representations
//!
//! Rotary Position Embeddings (RoPE), Attention with Linear Biases (ALiBi), learned absolute embeddings, and sinusoidal encodings.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

pub mod alibi;
pub mod learned;
pub mod rope;

use crate::config::PositionEncodingType;
use crate::core::TransformerResult;
use brain_core::Tensor;

/// Categorical kind of positional encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PositionKind {
    /// Rotary Position Embedding (RoPE).
    #[default]
    Rope,
    /// Attention with Linear Biases (ALiBi).
    Alibi,
    /// Vaswani et al. fixed sinusoidal encodings.
    Sinusoidal,
    /// Learned 1D embedding table.
    Learned,
    /// T5 / Shaw relative bias table.
    Relative,
    /// No positional encoding.
    None,
}

/// Unified trait for positional encoding layers.
pub trait PositionalEncoding: Send + Sync {
    /// Applies positional encoding to representation tensor `x` with given sequence offset.
    fn apply(&self, x: &Tensor, offset: usize) -> TransformerResult<Tensor>;

    /// Returns the positional encoding algorithm kind.
    fn kind(&self) -> PositionKind;
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
    fn test_position_registry_1() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }
}
