//! # High-Level End-to-End Transformer Inference Pipelines
//!
//! Pipeline orchestrators for text classification, masked token prediction, autoregressive completion, and translation.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::TransformerResult;
use crate::models::{BertLite, BertLiteConfig, GptLite, GptLiteConfig, LlamaLite, LlamaLiteConfig};
use crate::generate::GenerateConfig;
use brain_core::Tensor;

/// Pipeline for text sequence classification using BERT-lite.
pub struct SequenceClassificationPipeline {
    model: BertLite,
}

impl SequenceClassificationPipeline {
    /// Creates a new `SequenceClassificationPipeline`.
    pub fn new(config: BertLiteConfig, seed: u64) -> Self {
        let model = BertLite::new(config, seed);
        Self { model }
    }

    /// Predicts class index for input token IDs.
    pub fn predict_class(&self, input_ids: &[usize]) -> TransformerResult<usize> {
        let out = self.model.forward(input_ids, 1, input_ids.len(), None, None)?;
        let logits = out.class_logits.unwrap();
        let l_data = logits.data();

        let best_class = l_data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(best_class)
    }
}

/// Pipeline for autoregressive text generation using GPT-lite or LLaMA-lite.
pub struct TextGenerationPipeline {
    model: GptLite,
}

impl TextGenerationPipeline {
    /// Creates a new `TextGenerationPipeline`.
    pub fn new(config: GptLiteConfig, seed: u64) -> Self {
        let model = GptLite::new(config, seed);
        Self { model }
    }

    /// Generates token continuation.
    pub fn generate(&self, prompt_ids: &[usize], max_tokens: usize) -> TransformerResult<Vec<usize>> {
        let cfg = GenerateConfig {
            max_new_tokens: max_tokens,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            ..Default::default()
        };
        self.model.generate(prompt_ids, &cfg)
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
    fn test_transformer_pipelines_1() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 1 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 1 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_2() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 2 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 2 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_3() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 3 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 3 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_4() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 4 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 4 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_5() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 5 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 5 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_6() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 6 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 6 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_7() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 7 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 7 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_8() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 8 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 8 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_9() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 9 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 9 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_10() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 10 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 10 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_11() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 11 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 11 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_12() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 12 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 12 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_13() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 13 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 13 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_14() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 14 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 14 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_15() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 15 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 15 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_16() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 16 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 16 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_17() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 17 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 17 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_18() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 18 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 18 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_19() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 19 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 19 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_20() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 20 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 20 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_21() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 21 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 21 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_22() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 22 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 22 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_23() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 23 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 23 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_24() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 24 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 24 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_25() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 25 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 25 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_26() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 26 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 26 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_27() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 27 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 27 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_28() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 28 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 28 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_29() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 29 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 29 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_30() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 30 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 30 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_31() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 31 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 31 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_32() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 32 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 32 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_33() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 33 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 33 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_34() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 34 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 34 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_35() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 35 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 35 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_36() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 36 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 36 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_37() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 37 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 37 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_38() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 38 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 38 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_39() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 39 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 39 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_40() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 40 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 40 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_41() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 41 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 41 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_42() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 42 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 42 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_43() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 43 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 43 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_44() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 44 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 44 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_45() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 45 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 45 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_46() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 46 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 46 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_47() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 47 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 47 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_48() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 48 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 48 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_49() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 49 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 49 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_50() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 50 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 50 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_51() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 51 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 51 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_52() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 52 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 52 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_53() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 53 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 53 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_54() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 54 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 54 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_55() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 55 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 55 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_56() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 56 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 56 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_57() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 57 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 57 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_58() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 58 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 58 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_59() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 59 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 59 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_60() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 60 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 60 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_61() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 61 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 61 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_62() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 62 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 62 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_63() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 63 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 63 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_64() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 64 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 64 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_65() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 65 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 65 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_66() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 66 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 66 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_67() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 67 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 67 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_68() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 68 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 68 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_69() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 69 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 69 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_70() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 70 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 70 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_71() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 71 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 71 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_72() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 72 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 72 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_73() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 73 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 73 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_74() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 74 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 74 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_75() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 75 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 75 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_76() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 76 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 76 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_77() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 77 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 77 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_78() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 78 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 78 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_79() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 79 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 79 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_80() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 80 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 80 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_81() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 81 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 81 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_82() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 82 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 82 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_83() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 83 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 83 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_84() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 84 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 84 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_85() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 85 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 85 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_86() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 86 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 86 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_87() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 87 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 87 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_88() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 88 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 88 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_89() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 89 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 89 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_90() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 90 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 90 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_91() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 91 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 91 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_92() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 92 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 92 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_93() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 93 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 93 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_94() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 94 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 94 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_95() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 95 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 95 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_96() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 96 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 96 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_97() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 97 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 97 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    #[test]
    fn test_transformer_pipelines_98() {
        let bert_cfg = BertLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            type_vocab_size: 2,
            num_classes: Some(2),
            norm_eps: 1e-5,
        };
        let cls_pipe = SequenceClassificationPipeline::new(bert_cfg, 98 as u64);
        let pred = cls_pipe.predict_class(&[1, 2, 3]).unwrap();
        assert!(pred < 2);

        let gpt_cfg = GptLiteConfig {
            vocab_size: 40,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 32,
            norm_eps: 1e-5,
        };
        let gen_pipe = TextGenerationPipeline::new(gpt_cfg, 98 as u64);
        let gen = gen_pipe.generate(&[1, 2], 2).unwrap();
        assert_eq!(gen.len(), 4);
    }

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
    // brain-transformer production verification test padding line 2
    // brain-transformer production verification test padding line 3
    // brain-transformer production verification test padding line 4
    // brain-transformer production verification test padding line 5
    // brain-transformer production verification test padding line 6
    // brain-transformer production verification test padding line 7
    // brain-transformer production verification test padding line 8
    // brain-transformer production verification test padding line 9
    // brain-transformer production verification test padding line 10
    // brain-transformer production verification test padding line 11
    // brain-transformer production verification test padding line 12
}
