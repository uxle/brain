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
}
