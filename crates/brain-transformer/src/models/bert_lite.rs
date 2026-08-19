//! # BERT-Lite: Bidirectional Encoder Transformer Model
//!
//! Encoder-only architecture with token, positional, and segment embeddings, stacked bidirectional encoder layers, and `[CLS]` pooling.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::config::{ActivationType, NormPosition, NormType, PositionEncodingType};
use crate::core::{AttentionMask, TransformerResult};
use crate::embedding_layers::{EmbConfig, TransformerEmbedding};
use crate::encoder::{TransformerEncoder, TransformerEncoderConfig};
use crate::head::{ClsHead, HeadConfig, LmHead};
use brain_core::Tensor;

/// Configuration for BERT-lite model.
#[derive(Debug, Clone, PartialEq)]
pub struct BertLiteConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of encoder layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Intermediate expansion dimension for FFN.
    pub intermediate_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Token type / segment vocabulary size (default: 2).
    pub type_vocab_size: usize,
    /// Number of classification classes (optional).
    pub num_classes: Option<usize>,
    /// Normalization epsilon.
    pub norm_eps: f64,
}

impl Default for BertLiteConfig {
    fn default() -> Self {
        let hidden_dim = 768;
        let num_heads = 12;
        Self {
            vocab_size: 30522,
            hidden_dim,
            num_layers: 12,
            num_heads,
            head_dim: hidden_dim / num_heads,
            intermediate_dim: 3072,
            max_seq_len: 512,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-12,
        }
    }
}

/// Output container for BERT-lite forward inference.
#[derive(Debug, Clone)]
pub struct BertOutput {
    /// Last layer sequence representation `[batch_size, seq_len, hidden_dim]`.
    pub sequence_output: Tensor,
    /// Pooled `[CLS]` representation `[batch_size, hidden_dim]`.
    pub pooled_output: Option<Tensor>,
    /// Masked Language Modeling prediction logits `[batch_size, seq_len, vocab_size]`.
    pub mlm_logits: Option<Tensor>,
    /// Classification class logits `[batch_size, num_classes]`.
    pub class_logits: Option<Tensor>,
}

/// Production BERT-Lite Model.
#[derive(Debug, Clone)]
pub struct BertLite {
    /// Token + Position + Segment embedding layer.
    pub embeddings: TransformerEmbedding,
    /// Stacked bidirectional Transformer Encoder.
    pub encoder: TransformerEncoder,
    /// Classification head.
    pub pooler: ClsHead,
    /// Masked Language Modeling prediction head.
    pub mlm_head: LmHead,
    /// Configuration options.
    pub config: BertLiteConfig,
}

impl BertLite {
    /// Creates a new `BertLite` model.
    pub fn new(config: BertLiteConfig, seed: u64) -> Self {
        let emb_cfg = EmbConfig {
            vocab_size: config.vocab_size,
            hidden_dim: config.hidden_dim,
            max_position_embeddings: config.max_seq_len,
            type_vocab_size: Some(config.type_vocab_size),
            dropout: 0.1,
            pos_encoding: PositionEncodingType::Learned,
            norm_type: Some(NormType::LayerNorm),
            norm_eps: config.norm_eps,
        };
        let embeddings = TransformerEmbedding::new(emb_cfg, seed);

        let enc_cfg = TransformerEncoderConfig {
            num_layers: config.num_layers,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            intermediate_dim: config.intermediate_dim,
            norm_position: NormPosition::PostNorm,
            norm_type: NormType::LayerNorm,
            activation: ActivationType::Gelu,
            norm_eps: config.norm_eps,
        };
        let encoder = TransformerEncoder::new(enc_cfg, seed.wrapping_add(500));

        let head_cfg = HeadConfig {
            hidden_dim: config.hidden_dim,
            vocab_size: config.vocab_size,
            num_classes: config.num_classes,
            bias: true,
        };
        let pooler = ClsHead::new(head_cfg.clone(), seed.wrapping_add(1000));
        let mlm_head = LmHead::new(head_cfg, seed.wrapping_add(1500));

        Self {
            embeddings,
            encoder,
            pooler,
            mlm_head,
            config,
        }
    }

    /// Computes full forward pass of BERT model.
    pub fn forward(
        &self,
        input_ids: &[usize],
        batch_size: usize,
        seq_len: usize,
        token_type_ids: Option<&[usize]>,
        padding_mask: Option<&Tensor>,
    ) -> TransformerResult<BertOutput> {
        let emb = self.embeddings.forward(input_ids, batch_size, seq_len, token_type_ids, 0)?;

        let mask = if let Some(pad) = padding_mask {
            AttentionMask::Padding(pad.clone())
        } else {
            AttentionMask::None
        };

        let enc_out = self.encoder.forward(&emb, &mask, false)?;
        let class_logits = self.pooler.forward(&enc_out.last_hidden_state).ok();
        let mlm_logits = self.mlm_head.forward(&enc_out.last_hidden_state).ok();

        Ok(BertOutput {
            sequence_output: enc_out.last_hidden_state,
            pooled_output: None,
            mlm_logits,
            class_logits,
        })
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
    fn test_bert_lite_model_1() {
        let cfg = BertLiteConfig {
            vocab_size: 100,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(3),
            norm_eps: 1e-5,
        };
        let bert = BertLite::new(cfg, 1 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }
}
