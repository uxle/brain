//! # brain-transformer 🧠⚡
//!
//! Production-grade Transformer architectures for the **Brain** deep learning ecosystem in pure, safe Rust.
//!
//! ## Highlights
//! - **Attention Mechanisms**: Multi-Head Attention (MHA), Scaled Dot-Product, Relative Position (T5), FlashAttention-Lite ($O(1)$ memory block tiling), MQA/GQA, xFormers-Lite.
//! - **Positional Encodings**: Rotary Position Embeddings (RoPE 1D & 2D), Attention with Linear Biases (ALiBi), learned absolute embeddings, sinusoidal frequencies.
//! - **Normalizations & Activations**: Pre-LN & Post-LN LayerNorm, RMSNorm, GELU, SwiGLU, GEGLU, SiLU, QuickGELU, ReLU.
//! - **Model Architectures**: BERT-Lite (encoder-only), GPT-Lite (decoder-only), T5-Lite (encoder-decoder), LLaMA-Lite (RMSNorm, RoPE, SwiGLU, GQA).
//! - **Inference & Generation**: Multi-layer Key-Value (KV) Cache, greedy decoding, temperature sampling, Top-K, Top-P (nucleus), min-p, and repetition penalties.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

pub mod attention;
pub mod builder;
pub mod config;
pub mod core;
pub mod decoder;
pub mod embedding_layers;
pub mod encoder;
pub mod ffn;
pub mod generate;
pub mod head;
pub mod r#impl;
pub mod kv_cache;
pub mod models;
pub mod ops;
pub mod position;
pub mod utils;

pub use attention::{
    flash_lite::{FlashAttentionLite, FlashLiteConfig},
    make_attention,
    multi_head::{MhaConfig, MultiHeadAttention},
    multi_query::{repeat_kv, GqaConfig, GroupedQueryAttention, MqaConfig},
    relative::{RelativeAttention, RelativeConfig, RelativePositionBias},
    scaled::{scaled_dot_product_attention, SdpaConfig},
    xformers_lite::{XformersAttentionLite, XformersConfig},
    Attention, AttentionKind,
};

pub use builder::TransformerBuilder;
pub use config::{
    ActivationType, AttentionConfig, FfnConfig, ModelArch, NormPosition, NormType,
    PositionEncodingType, TransformerConfig,
};
pub use core::{AttentionMask, BatchSeq, LinearParams, TransformerError, TransformerResult};
pub use decoder::{
    cross::{CrossAttention, CrossAttnConfig},
    layer::{DecoderLayerConfig, TransformerDecoderLayer},
    DecoderConfig, DecoderOutput, TransformerDecoder,
};
pub use embedding_layers::{EmbConfig, TransformerEmbedding};
pub use encoder::{
    block::{BlockConfig, TransformerEncoderBlock},
    layer::EncoderLayer,
    EncoderOutput, TransformerEncoder, TransformerEncoderConfig,
};
pub use ffn::FeedForwardNetwork;
pub use generate::{GenerateConfig, Generator};
pub use head::{ClsHead, HeadConfig, LmHead};
pub use kv_cache::{KvCache, KvCacheConfig, LayerKvCache};
pub use models::{
    BertLite, BertLiteConfig, BertOutput, GptLite, GptLiteConfig, LlamaLite, LlamaLiteConfig,
    T5Lite, T5LiteConfig,
};
pub use ops::{
    apply_activation, apply_attention_mask, bmm, gelu, layer_norm, quick_gelu, relu, rms_norm,
    sigmoid, silu, softmax_2d, softmax_inplace,
};
pub use position::{
    alibi::{AlibiConfig, AlibiPositionalBias},
    learned::{LearnedPositionalEmbedding, PositionConfig, SinusoidalPositionalEmbedding},
    rope::{RopeConfig, RotaryEmbedding},
    PositionKind, PositionalEncoding,
};
pub use utils::{all_close, init_kaiming_normal, init_xavier_uniform, TransformerRng};

/// Convenient prelude module for `brain-transformer`.
pub mod prelude {
    pub use crate::attention::{
        flash_lite::FlashAttentionLite, make_attention, multi_head::MultiHeadAttention,
        multi_query::GroupedQueryAttention, relative::RelativeAttention,
        scaled::scaled_dot_product_attention, Attention, AttentionKind,
    };
    pub use crate::builder::TransformerBuilder;
    pub use crate::config::{
        ActivationType, AttentionConfig, FfnConfig, ModelArch, NormPosition, NormType,
        PositionEncodingType, TransformerConfig,
    };
    pub use crate::core::{
        AttentionMask, BatchSeq, LinearParams, TransformerError, TransformerResult,
    };
    pub use crate::decoder::{
        cross::CrossAttention, layer::TransformerDecoderLayer, DecoderConfig, DecoderOutput,
        TransformerDecoder,
    };
    pub use crate::embedding_layers::{EmbConfig, TransformerEmbedding};
    pub use crate::encoder::{
        block::TransformerEncoderBlock, layer::EncoderLayer, EncoderOutput, TransformerEncoder,
        TransformerEncoderConfig,
    };
    pub use crate::ffn::FeedForwardNetwork;
    pub use crate::generate::{GenerateConfig, Generator};
    pub use crate::head::{ClsHead, HeadConfig, LmHead};
    pub use crate::kv_cache::{KvCache, KvCacheConfig};
    pub use crate::models::{
        BertLite, BertLiteConfig, BertOutput, GptLite, GptLiteConfig, LlamaLite, LlamaLiteConfig,
        T5Lite, T5LiteConfig,
    };
    pub use crate::ops::{
        apply_activation, apply_attention_mask, bmm, gelu, layer_norm, quick_gelu, relu, rms_norm,
        sigmoid, silu, softmax_2d, softmax_inplace,
    };
    pub use crate::position::{
        alibi::AlibiPositionalBias,
        learned::{LearnedPositionalEmbedding, SinusoidalPositionalEmbedding},
        rope::RotaryEmbedding,
        PositionKind, PositionalEncoding,
    };
    pub use crate::utils::{all_close, TransformerRng};
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
    fn test_prelude_integration_1() {
        use crate::prelude::*;
        let cfg = TransformerConfig::default();
        assert!(cfg.validate().is_ok());

        let rng = TransformerRng::new(1 as u64);
        assert!(rng.clone().next_f64() < 1.0);

        let sin = SinusoidalPositionalEmbedding::generate(16, 8);
        assert_eq!(sin.shape(), &[16, 8]);
    }
}
