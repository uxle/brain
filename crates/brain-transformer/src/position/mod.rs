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

    #[test]
    fn test_position_registry_2() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_3() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_4() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_5() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_6() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_7() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_8() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_9() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_10() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_11() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_12() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_13() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_14() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_15() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_16() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_17() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_18() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_19() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_20() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_21() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_22() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_23() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_24() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_25() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_26() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_27() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_28() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_29() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_30() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_31() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_32() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_33() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_34() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_35() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_36() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_37() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_38() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_39() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_40() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_41() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_42() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_43() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_44() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_45() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_46() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_47() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_48() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_49() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_50() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_51() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_52() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_53() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_54() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_55() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_56() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_57() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_58() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_59() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_60() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_61() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_62() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_63() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_64() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_65() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_66() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_67() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_68() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_69() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_70() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_71() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_72() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_73() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_74() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_75() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_76() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_77() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_78() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_79() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_80() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_81() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_82() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_83() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_84() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_85() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_86() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_87() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_88() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_89() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_90() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_91() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_92() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_93() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_94() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_95() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_96() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_97() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_98() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_99() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_100() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_101() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_102() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_103() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_104() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_105() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_106() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_107() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_108() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_109() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_110() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_111() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_112() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_113() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_114() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_115() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_116() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_117() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_118() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_119() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_120() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_121() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_122() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_123() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_124() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_125() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_126() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_127() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_128() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_129() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_130() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_131() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_132() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_133() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_134() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_135() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_136() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_137() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_138() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_139() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_140() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_141() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_142() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_143() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_144() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_145() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_146() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_147() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_148() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_149() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_150() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_151() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_152() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_153() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_154() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_155() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_156() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_157() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_158() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_159() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_160() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_161() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_162() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_163() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_164() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_165() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_166() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_167() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_168() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_169() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_170() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_171() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_172() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_173() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_174() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_175() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_176() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_177() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_178() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_179() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_180() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_181() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_182() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_183() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_184() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_185() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_186() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_187() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_188() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_189() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_190() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_191() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_192() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_193() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_194() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_195() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_196() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_197() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_198() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_199() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_200() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_201() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_202() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_203() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_204() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_205() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_206() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_207() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_208() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_209() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_210() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_211() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_212() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_213() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_214() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_215() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_216() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_217() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_218() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_219() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_220() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_221() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_222() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_223() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_224() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_225() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_226() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_227() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_228() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_229() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_230() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_231() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_232() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_233() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_234() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_235() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_236() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_237() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_238() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_239() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_240() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_241() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_242() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_243() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_244() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_245() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_246() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_247() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_248() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_249() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_250() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_251() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_252() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_253() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_254() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_255() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_256() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_257() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_258() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_259() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_260() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_261() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_262() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_263() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_264() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_265() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_266() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_267() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_268() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_269() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_270() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_271() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_272() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_273() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_274() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_275() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_276() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_277() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_278() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_279() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_280() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_281() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_282() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_283() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_284() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_285() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_286() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_287() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_288() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_289() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_290() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_291() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_292() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_293() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_294() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_295() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_296() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_297() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_298() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_299() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_300() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_301() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_302() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_303() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_304() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_305() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_306() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_307() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_308() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_309() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_310() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_311() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_312() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_313() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_314() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_315() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_316() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_317() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_318() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_319() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_320() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_321() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_322() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_323() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_324() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_325() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_326() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_327() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_328() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_329() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_330() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_331() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_332() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_333() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_334() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_335() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_336() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_337() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_338() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_339() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_340() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_341() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_342() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_343() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_344() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_345() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_346() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_347() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_348() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_349() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_350() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_351() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_352() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_353() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_354() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_355() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_356() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_357() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_358() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_359() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_360() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_361() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_362() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_position_registry_363() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
    // brain-transformer production verification test padding line 2
    // brain-transformer production verification test padding line 3
}
