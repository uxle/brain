//! # Fluent Builder API for Transformer Models and Stacks
//!
//! Ergonomic fluent builders for configuring and instantiating BERT, GPT, T5, LLaMA, and custom transformer architectures.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::config::{
    ActivationType, AttentionConfig, FfnConfig, ModelArch, NormPosition, NormType,
    PositionEncodingType, TransformerConfig,
};
use crate::core::TransformerResult;
use crate::decoder::{DecoderConfig, TransformerDecoder};
use crate::encoder::{TransformerEncoder, TransformerEncoderConfig};

/// Fluent builder for constructing `TransformerConfig` and transformer modules.
#[derive(Debug, Clone)]
pub struct TransformerBuilder {
    config: TransformerConfig,
}

impl TransformerBuilder {
    /// Creates a new `TransformerBuilder` with default configurations.
    pub fn new() -> Self {
        Self {
            config: TransformerConfig::default(),
        }
    }

    /// Sets model architecture family (EncoderOnly, DecoderOnly, EncoderDecoder).
    pub fn arch(mut self, arch: ModelArch) -> Self {
        self.config.arch = arch;
        self
    }

    /// Sets vocabulary size.
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Sets hidden model dimension $d_{\text{model}}$.
    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.config.hidden_dim = dim;
        self.config.attention.hidden_dim = dim;
        self.config.ffn.hidden_dim = dim;
        self
    }

    /// Sets number of transformer layers.
    pub fn num_layers(mut self, layers: usize) -> Self {
        self.config.num_layers = layers;
        self
    }

    /// Sets number of attention heads.
    pub fn num_heads(mut self, heads: usize) -> Self {
        self.config.attention.num_heads = heads;
        self.config.attention.num_kv_heads = heads;
        if self.config.hidden_dim > 0 && heads > 0 {
            self.config.attention.head_dim = self.config.hidden_dim / heads;
        }
        self
    }

    /// Sets FFN intermediate expansion dimension.
    pub fn intermediate_dim(mut self, dim: usize) -> Self {
        self.config.ffn.intermediate_dim = dim;
        self
    }

    /// Sets non-linear activation type.
    pub fn activation(mut self, act: ActivationType) -> Self {
        self.config.ffn.activation = act;
        self
    }

    /// Sets normalization placement (PreNorm / PostNorm).
    pub fn norm_position(mut self, pos: NormPosition) -> Self {
        self.config.norm_position = pos;
        self
    }

    /// Sets normalization algorithm (LayerNorm / RMSNorm).
    pub fn norm_type(mut self, n_type: NormType) -> Self {
        self.config.norm_type = n_type;
        self
    }

    /// Sets positional encoding strategy (RoPE, ALiBi, Sinusoidal, Learned).
    pub fn pos_encoding(mut self, pe: PositionEncodingType) -> Self {
        self.config.attention.pos_encoding = pe;
        self
    }

    /// Sets maximum context sequence length.
    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.config.max_seq_len = len;
        self.config.attention.max_seq_len = len;
        self
    }

    /// Builds and validates the configuration.
    pub fn build_config(self) -> TransformerResult<TransformerConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Instantiates a `TransformerEncoder` from builder parameters.
    pub fn build_encoder(self, seed: u64) -> TransformerResult<TransformerEncoder> {
        self.config.validate()?;
        let enc_cfg = TransformerEncoderConfig {
            num_layers: self.config.num_layers,
            hidden_dim: self.config.hidden_dim,
            num_heads: self.config.attention.num_heads,
            head_dim: self.config.attention.head_dim,
            intermediate_dim: self.config.ffn.intermediate_dim,
            norm_position: self.config.norm_position,
            norm_type: self.config.norm_type,
            activation: self.config.ffn.activation,
            norm_eps: self.config.norm_eps,
        };
        Ok(TransformerEncoder::new(enc_cfg, seed))
    }

    /// Instantiates a `TransformerDecoder` from builder parameters.
    pub fn build_decoder(self, seed: u64) -> TransformerResult<TransformerDecoder> {
        self.config.validate()?;
        let dec_cfg = DecoderConfig {
            num_layers: self.config.num_layers,
            hidden_dim: self.config.hidden_dim,
            num_heads: self.config.attention.num_heads,
            head_dim: self.config.attention.head_dim,
            intermediate_dim: self.config.ffn.intermediate_dim,
            has_cross_attention: self.config.arch == ModelArch::EncoderDecoder,
            norm_position: self.config.norm_position,
            norm_type: self.config.norm_type,
            activation: self.config.ffn.activation,
            norm_eps: self.config.norm_eps,
        };
        Ok(TransformerDecoder::new(dec_cfg, seed))
    }
}

impl Default for TransformerBuilder {
    fn default() -> Self {
        Self::new()
    }
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
    fn test_transformer_builder_1() {
        let b = TransformerBuilder::new()
            .vocab_size(5000)
            .hidden_dim(32)
            .num_heads(4)
            .num_layers(2)
            .intermediate_dim(64)
            .activation(ActivationType::Gelu)
            .norm_type(NormType::LayerNorm)
            .max_seq_len(128);

        let cfg = b.clone().build_config().unwrap();
        assert_eq!(cfg.vocab_size, 5000);
        assert_eq!(cfg.hidden_dim, 32);

        let enc = b.clone().build_encoder(1 as u64).unwrap();
        assert_eq!(enc.layers.len(), 2);

        let dec = b.build_decoder(1 as u64).unwrap();
        assert_eq!(dec.layers.len(), 2);
    }
}
