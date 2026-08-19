//! # Transformer Prediction Heads & Task Adapters
//!
//! Language Modeling heads (`LmHead`), Classification poolers (`ClsHead`), and Seq2Seq output projection heads.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{LinearParams, TransformerError, TransformerResult};
use brain_core::Tensor;

/// Configuration for transformer prediction heads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadConfig {
    /// Input hidden representation dimension.
    pub hidden_dim: usize,
    /// Vocabulary size for language modeling.
    pub vocab_size: usize,
    /// Number of target classes for classification.
    pub num_classes: Option<usize>,
    /// Include bias vector in head projection.
    pub bias: bool,
}

impl Default for HeadConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            vocab_size: 32000,
            num_classes: None,
            bias: false,
        }
    }
}

/// Language Modeling Head projecting hidden states to vocabulary logits: $Z = X W_{\text{lm}} + b$.
#[derive(Debug, Clone)]
pub struct LmHead {
    /// Linear projection parameters `[hidden_dim, vocab_size]`.
    pub proj: LinearParams,
    /// Configuration options.
    pub config: HeadConfig,
}

impl LmHead {
    /// Creates a new `LmHead`.
    pub fn new(config: HeadConfig, seed: u64) -> Self {
        let proj = LinearParams::new(config.hidden_dim, config.vocab_size, config.bias, seed);
        Self { proj, config }
    }

    /// Creates an `LmHead` with tied weights shared with token embedding table `[vocab_size, hidden_dim]`.
    pub fn new_tied(config: HeadConfig, embedding_weights: &Tensor) -> Self {
        // Transpose embedding weights: [vocab_size, hidden_dim] -> [hidden_dim, vocab_size]
        let v_size = config.vocab_size;
        let h_dim = config.hidden_dim;
        let emb_data = embedding_weights.data();
        let mut tied_data = vec![0.0f64; h_dim * v_size];

        for v in 0..v_size {
            for h in 0..h_dim {
                tied_data[h * v_size + v] = emb_data[v * h_dim + h];
            }
        }

        let weight = Tensor::from_vec(tied_data, vec![h_dim, v_size]);
        let proj = LinearParams {
            weight,
            bias: if config.bias { Some(Tensor::zeros(vec![v_size])) } else { None },
            in_features: h_dim,
            out_features: v_size,
        };

        Self { proj, config }
    }

    /// Projects hidden states to vocabulary logits `[batch_size, seq_len, vocab_size]`.
    pub fn forward(&self, hidden_states: &Tensor) -> TransformerResult<Tensor> {
        self.proj.forward(hidden_states)
    }
}

/// Sequence Classification Head with `[CLS]` token extraction and projection.
#[derive(Debug, Clone)]
pub struct ClsHead {
    /// Dense pooling projection `[hidden_dim, hidden_dim]`.
    pub dense: LinearParams,
    /// Final classifier projection `[hidden_dim, num_classes]`.
    pub classifier: LinearParams,
    /// Configuration options.
    pub config: HeadConfig,
}

impl ClsHead {
    /// Creates a new `ClsHead`.
    pub fn new(config: HeadConfig, seed: u64) -> Self {
        let num_classes = config.num_classes.unwrap_or(2);
        let dense = LinearParams::new(config.hidden_dim, config.hidden_dim, true, seed);
        let classifier = LinearParams::new(config.hidden_dim, num_classes, true, seed.wrapping_add(100));

        Self {
            dense,
            classifier,
            config,
        }
    }

    /// Extracts first token `[CLS]` (at index 0) from sequence `[batch_size, seq_len, hidden_dim]` and computes class logits `[batch_size, num_classes]`.
    pub fn forward(&self, hidden_states: &Tensor) -> TransformerResult<Tensor> {
        let shape = hidden_states.shape();
        if shape.len() != 3 {
            return Err(TransformerError::DimensionMismatch {
                expected: 3,
                found: shape.len(),
            });
        }

        let batch_size = shape[0];
        let seq_len = shape[1];
        let hidden_dim = shape[2];

        let in_data = hidden_states.data();
        let mut cls_data = Vec::with_capacity(batch_size * hidden_dim);

        for b in 0..batch_size {
            let offset = b * seq_len * hidden_dim;
            cls_data.extend_from_slice(&in_data[offset..offset + hidden_dim]);
        }

        let cls_tensor = Tensor::from_vec(cls_data, vec![batch_size, hidden_dim]);
        let pooled = self.dense.forward(&cls_tensor)?;

        // Apply tanh activation on pooled state
        let mut pooled_data = pooled.data().to_vec();
        for x in pooled_data.iter_mut() {
            *x = x.tanh();
        }
        let pooled_act = Tensor::from_vec(pooled_data, vec![batch_size, hidden_dim]);

        self.classifier.forward(&pooled_act)
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
    fn test_transformer_heads_1() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 1 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }
}
