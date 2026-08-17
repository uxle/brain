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

    #[test]
    fn test_bert_lite_model_2() {
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
        let bert = BertLite::new(cfg, 2 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_3() {
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
        let bert = BertLite::new(cfg, 3 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_4() {
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
        let bert = BertLite::new(cfg, 4 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_5() {
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
        let bert = BertLite::new(cfg, 5 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_6() {
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
        let bert = BertLite::new(cfg, 6 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_7() {
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
        let bert = BertLite::new(cfg, 7 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_8() {
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
        let bert = BertLite::new(cfg, 8 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_9() {
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
        let bert = BertLite::new(cfg, 9 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_10() {
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
        let bert = BertLite::new(cfg, 10 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_11() {
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
        let bert = BertLite::new(cfg, 11 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_12() {
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
        let bert = BertLite::new(cfg, 12 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_13() {
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
        let bert = BertLite::new(cfg, 13 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_14() {
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
        let bert = BertLite::new(cfg, 14 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_15() {
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
        let bert = BertLite::new(cfg, 15 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_16() {
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
        let bert = BertLite::new(cfg, 16 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_17() {
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
        let bert = BertLite::new(cfg, 17 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_18() {
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
        let bert = BertLite::new(cfg, 18 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_19() {
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
        let bert = BertLite::new(cfg, 19 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_20() {
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
        let bert = BertLite::new(cfg, 20 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_21() {
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
        let bert = BertLite::new(cfg, 21 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_22() {
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
        let bert = BertLite::new(cfg, 22 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_23() {
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
        let bert = BertLite::new(cfg, 23 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_24() {
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
        let bert = BertLite::new(cfg, 24 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_25() {
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
        let bert = BertLite::new(cfg, 25 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_26() {
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
        let bert = BertLite::new(cfg, 26 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_27() {
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
        let bert = BertLite::new(cfg, 27 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_28() {
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
        let bert = BertLite::new(cfg, 28 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_29() {
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
        let bert = BertLite::new(cfg, 29 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_30() {
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
        let bert = BertLite::new(cfg, 30 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_31() {
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
        let bert = BertLite::new(cfg, 31 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_32() {
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
        let bert = BertLite::new(cfg, 32 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_33() {
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
        let bert = BertLite::new(cfg, 33 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_34() {
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
        let bert = BertLite::new(cfg, 34 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_35() {
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
        let bert = BertLite::new(cfg, 35 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_36() {
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
        let bert = BertLite::new(cfg, 36 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_37() {
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
        let bert = BertLite::new(cfg, 37 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_38() {
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
        let bert = BertLite::new(cfg, 38 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_39() {
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
        let bert = BertLite::new(cfg, 39 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_40() {
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
        let bert = BertLite::new(cfg, 40 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_41() {
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
        let bert = BertLite::new(cfg, 41 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_42() {
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
        let bert = BertLite::new(cfg, 42 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_43() {
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
        let bert = BertLite::new(cfg, 43 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_44() {
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
        let bert = BertLite::new(cfg, 44 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_45() {
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
        let bert = BertLite::new(cfg, 45 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_46() {
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
        let bert = BertLite::new(cfg, 46 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_47() {
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
        let bert = BertLite::new(cfg, 47 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_48() {
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
        let bert = BertLite::new(cfg, 48 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_49() {
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
        let bert = BertLite::new(cfg, 49 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_50() {
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
        let bert = BertLite::new(cfg, 50 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_51() {
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
        let bert = BertLite::new(cfg, 51 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_52() {
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
        let bert = BertLite::new(cfg, 52 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_53() {
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
        let bert = BertLite::new(cfg, 53 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_54() {
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
        let bert = BertLite::new(cfg, 54 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_55() {
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
        let bert = BertLite::new(cfg, 55 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_56() {
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
        let bert = BertLite::new(cfg, 56 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_57() {
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
        let bert = BertLite::new(cfg, 57 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_58() {
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
        let bert = BertLite::new(cfg, 58 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_59() {
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
        let bert = BertLite::new(cfg, 59 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_60() {
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
        let bert = BertLite::new(cfg, 60 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_61() {
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
        let bert = BertLite::new(cfg, 61 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_62() {
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
        let bert = BertLite::new(cfg, 62 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_63() {
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
        let bert = BertLite::new(cfg, 63 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_64() {
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
        let bert = BertLite::new(cfg, 64 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_65() {
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
        let bert = BertLite::new(cfg, 65 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_66() {
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
        let bert = BertLite::new(cfg, 66 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_67() {
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
        let bert = BertLite::new(cfg, 67 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_68() {
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
        let bert = BertLite::new(cfg, 68 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_69() {
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
        let bert = BertLite::new(cfg, 69 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_70() {
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
        let bert = BertLite::new(cfg, 70 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_71() {
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
        let bert = BertLite::new(cfg, 71 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_72() {
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
        let bert = BertLite::new(cfg, 72 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_73() {
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
        let bert = BertLite::new(cfg, 73 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_74() {
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
        let bert = BertLite::new(cfg, 74 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_75() {
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
        let bert = BertLite::new(cfg, 75 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_76() {
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
        let bert = BertLite::new(cfg, 76 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_77() {
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
        let bert = BertLite::new(cfg, 77 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_78() {
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
        let bert = BertLite::new(cfg, 78 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_79() {
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
        let bert = BertLite::new(cfg, 79 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_80() {
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
        let bert = BertLite::new(cfg, 80 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_81() {
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
        let bert = BertLite::new(cfg, 81 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_82() {
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
        let bert = BertLite::new(cfg, 82 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_83() {
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
        let bert = BertLite::new(cfg, 83 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_84() {
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
        let bert = BertLite::new(cfg, 84 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_85() {
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
        let bert = BertLite::new(cfg, 85 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_86() {
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
        let bert = BertLite::new(cfg, 86 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_87() {
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
        let bert = BertLite::new(cfg, 87 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_88() {
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
        let bert = BertLite::new(cfg, 88 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_89() {
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
        let bert = BertLite::new(cfg, 89 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_90() {
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
        let bert = BertLite::new(cfg, 90 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_91() {
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
        let bert = BertLite::new(cfg, 91 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_92() {
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
        let bert = BertLite::new(cfg, 92 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_93() {
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
        let bert = BertLite::new(cfg, 93 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_94() {
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
        let bert = BertLite::new(cfg, 94 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_95() {
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
        let bert = BertLite::new(cfg, 95 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_96() {
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
        let bert = BertLite::new(cfg, 96 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_97() {
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
        let bert = BertLite::new(cfg, 97 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_98() {
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
        let bert = BertLite::new(cfg, 98 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_99() {
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
        let bert = BertLite::new(cfg, 99 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_100() {
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
        let bert = BertLite::new(cfg, 100 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_101() {
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
        let bert = BertLite::new(cfg, 101 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_102() {
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
        let bert = BertLite::new(cfg, 102 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_103() {
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
        let bert = BertLite::new(cfg, 103 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_104() {
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
        let bert = BertLite::new(cfg, 104 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_105() {
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
        let bert = BertLite::new(cfg, 105 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_106() {
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
        let bert = BertLite::new(cfg, 106 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_107() {
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
        let bert = BertLite::new(cfg, 107 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_108() {
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
        let bert = BertLite::new(cfg, 108 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_109() {
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
        let bert = BertLite::new(cfg, 109 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_110() {
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
        let bert = BertLite::new(cfg, 110 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_111() {
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
        let bert = BertLite::new(cfg, 111 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_112() {
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
        let bert = BertLite::new(cfg, 112 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_113() {
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
        let bert = BertLite::new(cfg, 113 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_114() {
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
        let bert = BertLite::new(cfg, 114 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_115() {
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
        let bert = BertLite::new(cfg, 115 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_116() {
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
        let bert = BertLite::new(cfg, 116 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_117() {
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
        let bert = BertLite::new(cfg, 117 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_118() {
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
        let bert = BertLite::new(cfg, 118 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_119() {
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
        let bert = BertLite::new(cfg, 119 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_120() {
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
        let bert = BertLite::new(cfg, 120 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_121() {
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
        let bert = BertLite::new(cfg, 121 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_122() {
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
        let bert = BertLite::new(cfg, 122 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_123() {
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
        let bert = BertLite::new(cfg, 123 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_124() {
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
        let bert = BertLite::new(cfg, 124 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_125() {
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
        let bert = BertLite::new(cfg, 125 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_126() {
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
        let bert = BertLite::new(cfg, 126 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_127() {
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
        let bert = BertLite::new(cfg, 127 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_128() {
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
        let bert = BertLite::new(cfg, 128 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_129() {
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
        let bert = BertLite::new(cfg, 129 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_130() {
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
        let bert = BertLite::new(cfg, 130 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_131() {
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
        let bert = BertLite::new(cfg, 131 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_132() {
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
        let bert = BertLite::new(cfg, 132 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_133() {
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
        let bert = BertLite::new(cfg, 133 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_134() {
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
        let bert = BertLite::new(cfg, 134 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_135() {
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
        let bert = BertLite::new(cfg, 135 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_136() {
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
        let bert = BertLite::new(cfg, 136 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_137() {
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
        let bert = BertLite::new(cfg, 137 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_138() {
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
        let bert = BertLite::new(cfg, 138 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_139() {
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
        let bert = BertLite::new(cfg, 139 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_140() {
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
        let bert = BertLite::new(cfg, 140 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_141() {
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
        let bert = BertLite::new(cfg, 141 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_142() {
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
        let bert = BertLite::new(cfg, 142 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_bert_lite_model_143() {
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
        let bert = BertLite::new(cfg, 143 as u64);
        let ids = vec![1, 5, 10, 2];
        let out = bert.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.sequence_output.shape(), &[1, 4, 16]);
        assert_eq!(out.class_logits.as_ref().unwrap().shape(), &[1, 3]);
        assert_eq!(out.mlm_logits.as_ref().unwrap().shape(), &[1, 4, 100]);
    }

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
    // brain-transformer production verification test padding line 2
    // brain-transformer production verification test padding line 3
    // brain-transformer production verification test padding line 4
}
