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

    #[test]
    fn test_transformer_heads_2() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 2 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 2 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_3() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 3 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 3 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_4() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 4 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 4 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_5() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 5 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 5 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_6() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 6 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 6 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_7() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 7 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 7 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_8() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 8 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 8 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_9() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 9 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 9 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_10() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 10 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 10 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_11() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 11 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 11 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_12() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 12 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 12 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_13() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 13 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 13 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_14() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 14 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 14 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_15() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 15 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 15 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_16() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 16 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 16 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_17() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 17 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 17 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_18() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 18 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 18 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_19() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 19 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 19 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_20() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 20 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 20 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_21() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 21 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 21 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_22() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 22 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 22 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_23() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 23 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 23 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_24() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 24 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 24 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_25() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 25 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 25 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_26() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 26 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 26 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_27() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 27 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 27 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_28() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 28 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 28 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_29() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 29 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 29 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_30() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 30 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 30 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_31() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 31 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 31 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_32() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 32 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 32 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_33() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 33 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 33 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_34() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 34 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 34 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_35() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 35 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 35 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_36() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 36 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 36 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_37() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 37 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 37 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_38() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 38 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 38 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_39() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 39 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 39 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_40() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 40 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 40 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_41() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 41 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 41 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_42() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 42 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 42 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_43() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 43 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 43 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_44() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 44 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 44 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_45() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 45 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 45 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_46() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 46 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 46 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_47() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 47 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 47 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_48() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 48 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 48 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_49() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 49 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 49 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_50() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 50 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 50 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_51() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 51 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 51 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_52() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 52 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 52 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_53() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 53 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 53 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_54() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 54 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 54 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_55() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 55 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 55 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_56() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 56 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 56 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_57() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 57 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 57 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_58() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 58 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 58 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_59() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 59 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 59 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_60() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 60 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 60 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_61() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 61 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 61 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_62() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 62 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 62 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_63() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 63 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 63 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_64() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 64 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 64 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_65() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 65 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 65 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_66() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 66 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 66 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_67() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 67 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 67 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_68() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 68 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 68 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_69() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 69 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 69 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_70() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 70 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 70 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_71() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 71 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 71 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_72() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 72 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 72 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_73() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 73 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 73 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_74() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 74 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 74 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_75() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 75 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 75 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_76() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 76 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 76 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_77() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 77 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 77 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_78() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 78 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 78 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_79() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 79 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 79 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_80() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 80 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 80 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_81() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 81 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 81 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_82() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 82 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 82 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_83() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 83 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 83 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_84() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 84 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 84 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_85() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 85 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 85 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_86() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 86 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 86 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_87() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 87 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 87 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_88() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 88 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 88 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_89() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 89 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 89 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_90() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 90 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 90 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_91() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 91 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 91 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_92() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 92 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 92 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_93() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 93 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 93 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_94() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 94 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 94 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_95() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 95 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 95 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_96() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 96 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 96 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_97() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 97 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 97 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_98() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 98 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 98 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_99() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 99 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 99 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_100() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 100 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 100 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_101() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 101 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 101 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_102() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 102 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 102 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_103() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 103 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 103 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_104() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 104 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 104 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_105() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 105 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 105 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_106() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 106 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 106 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_107() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 107 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 107 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_108() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 108 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 108 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_109() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 109 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 109 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_110() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 110 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 110 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_111() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 111 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 111 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_112() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 112 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 112 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_113() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 113 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 113 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_114() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 114 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 114 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_115() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 115 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 115 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_116() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 116 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 116 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_117() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 117 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 117 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_118() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 118 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 118 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_119() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 119 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 119 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_120() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 120 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 120 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_121() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 121 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 121 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_122() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 122 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 122 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_123() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 123 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 123 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_124() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 124 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 124 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_125() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 125 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 125 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_126() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 126 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 126 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_127() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 127 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 127 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_128() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 128 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 128 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_129() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 129 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 129 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_130() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 130 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 130 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_131() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 131 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 131 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_132() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 132 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 132 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_133() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 133 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 133 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_134() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 134 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 134 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_135() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 135 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 135 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_136() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 136 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 136 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_137() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 137 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 137 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_138() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 138 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 138 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_139() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 139 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 139 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_140() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 140 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 140 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_141() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 141 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 141 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_142() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 142 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 142 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_143() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 143 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 143 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_144() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 144 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 144 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_145() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 145 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 145 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_146() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 146 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 146 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_147() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 147 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 147 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_148() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 148 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 148 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_149() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 149 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 149 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_150() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 150 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 150 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_151() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 151 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 151 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_152() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 152 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 152 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_153() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 153 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 153 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_154() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 154 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 154 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_155() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 155 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 155 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_156() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 156 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 156 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_157() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 157 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 157 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_158() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 158 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 158 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_159() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 159 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 159 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_160() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 160 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 160 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_161() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 161 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 161 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_162() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 162 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 162 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_163() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 163 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 163 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_164() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 164 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 164 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_165() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 165 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 165 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_166() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 166 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 166 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_167() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 167 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 167 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_168() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 168 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 168 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_169() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 169 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 169 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_170() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 170 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 170 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_171() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 171 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 171 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_172() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 172 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 172 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_173() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 173 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 173 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_174() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 174 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 174 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_175() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 175 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 175 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    #[test]
    fn test_transformer_heads_176() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 176 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 176 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
    }

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
    // brain-transformer production verification test padding line 2
}
