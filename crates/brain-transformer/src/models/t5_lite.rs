//! # T5-Lite: Encoder-Decoder Sequence-to-Sequence Model
//!
//! Sequence-to-Sequence transformer with shared vocabulary embeddings, relative position biases, and cross-attention.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::attention::relative::RelativeConfig;
use crate::config::{ActivationType, NormPosition, NormType, PositionEncodingType};
use crate::core::{AttentionMask, TransformerResult};
use crate::decoder::{DecoderConfig, TransformerDecoder};
use crate::embedding_layers::{EmbConfig, TransformerEmbedding};
use crate::encoder::{TransformerEncoder, TransformerEncoderConfig};
use crate::head::{HeadConfig, LmHead};
use brain_core::Tensor;

/// Configuration for T5-lite model.
#[derive(Debug, Clone, PartialEq)]
pub struct T5LiteConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of encoder layers.
    pub num_encoder_layers: usize,
    /// Number of decoder layers.
    pub num_decoder_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Intermediate expansion dimension for FFN.
    pub intermediate_dim: usize,
    /// Number of relative position buckets.
    pub num_buckets: usize,
    /// Maximum distance threshold for relative bias.
    pub max_distance: usize,
    /// Normalization epsilon.
    pub norm_eps: f64,
}

impl Default for T5LiteConfig {
    fn default() -> Self {
        let hidden_dim = 768;
        let num_heads = 12;
        Self {
            vocab_size: 32128,
            hidden_dim,
            num_encoder_layers: 12,
            num_decoder_layers: 12,
            num_heads,
            head_dim: hidden_dim / num_heads,
            intermediate_dim: 3072,
            num_buckets: 32,
            max_distance: 128,
            norm_eps: 1e-6,
        }
    }
}

/// Production T5-Lite Encoder-Decoder Model.
#[derive(Debug, Clone)]
pub struct T5Lite {
    /// Shared word embeddings.
    pub shared_embeddings: TransformerEmbedding,
    /// Transformer Encoder stack.
    pub encoder: TransformerEncoder,
    /// Transformer Decoder stack.
    pub decoder: TransformerDecoder,
    /// Target LM projection head.
    pub lm_head: LmHead,
    /// Configuration options.
    pub config: T5LiteConfig,
}

impl T5Lite {
    /// Creates a new `T5Lite` model.
    pub fn new(config: T5LiteConfig, seed: u64) -> Self {
        let emb_cfg = EmbConfig {
            vocab_size: config.vocab_size,
            hidden_dim: config.hidden_dim,
            max_position_embeddings: 512,
            type_vocab_size: None,
            dropout: 0.0,
            pos_encoding: PositionEncodingType::None,
            norm_type: None,
            norm_eps: config.norm_eps,
        };
        let shared_embeddings = TransformerEmbedding::new(emb_cfg, seed);

        let enc_cfg = TransformerEncoderConfig {
            num_layers: config.num_encoder_layers,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            intermediate_dim: config.intermediate_dim,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: config.norm_eps,
        };
        let encoder = TransformerEncoder::new(enc_cfg, seed.wrapping_add(500));

        let dec_cfg = DecoderConfig {
            num_layers: config.num_decoder_layers,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            intermediate_dim: config.intermediate_dim,
            has_cross_attention: true,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: config.norm_eps,
        };
        let decoder = TransformerDecoder::new(dec_cfg, seed.wrapping_add(1000));

        let head_cfg = HeadConfig {
            hidden_dim: config.hidden_dim,
            vocab_size: config.vocab_size,
            num_classes: None,
            bias: false,
        };
        let lm_head = LmHead::new(head_cfg, seed.wrapping_add(1500));

        Self {
            shared_embeddings,
            encoder,
            decoder,
            lm_head,
            config,
        }
    }

    /// Computes full encoder-decoder forward pass.
    pub fn forward(
        &self,
        encoder_input_ids: &[usize],
        decoder_input_ids: &[usize],
        batch_size: usize,
        enc_len: usize,
        dec_len: usize,
    ) -> TransformerResult<Tensor> {
        let enc_emb =
            self.shared_embeddings
                .forward(encoder_input_ids, batch_size, enc_len, None, 0)?;
        let enc_out = self
            .encoder
            .forward(&enc_emb, &AttentionMask::None, false)?;

        let dec_emb =
            self.shared_embeddings
                .forward(decoder_input_ids, batch_size, dec_len, None, 0)?;
        let dec_out = self.decoder.forward(
            &dec_emb,
            Some(&enc_out.last_hidden_state),
            &AttentionMask::Causal,
            &AttentionMask::None,
            false,
        )?;

        self.lm_head.forward(&dec_out.last_hidden_state)
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
    fn test_t5_lite_model_1() {
        let cfg = T5LiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            num_buckets: 16,
            max_distance: 64,
            norm_eps: 1e-5,
        };
        let t5 = T5Lite::new(cfg, 1 as u64);
        let enc_ids = vec![1, 2, 3];
        let dec_ids = vec![10, 11];

        let out = t5.forward(&enc_ids, &dec_ids, 1, 3, 2).unwrap();
        assert_eq!(out.shape(), &[1, 2, 40]);
    }
}
