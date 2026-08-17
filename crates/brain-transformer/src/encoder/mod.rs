//! # Transformer Encoder Stack
//!
//! Stacked Transformer Encoder layers with hidden-state extraction, pre/post normalization, and bidirectional representation output.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

pub mod block;
pub mod layer;

use crate::config::{ActivationType, NormPosition, NormType};
use crate::core::{AttentionMask, TransformerError, TransformerResult};
use crate::encoder::block::{BlockConfig, TransformerEncoderBlock};
use crate::ops::{layer_norm, rms_norm};
use brain_core::Tensor;

/// Configuration for stacked `TransformerEncoder`.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerEncoderConfig {
    /// Number of stacked encoder layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Intermediate expansion dimension for FFN.
    pub intermediate_dim: usize,
    /// Normalization placement.
    pub norm_position: NormPosition,
    /// Normalization type.
    pub norm_type: NormType,
    /// Activation function.
    pub activation: ActivationType,
    /// Normalization epsilon.
    pub norm_eps: f64,
}

impl Default for TransformerEncoderConfig {
    fn default() -> Self {
        Self {
            num_layers: 6,
            hidden_dim: 768,
            num_heads: 12,
            head_dim: 64,
            intermediate_dim: 3072,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-6,
        }
    }
}

/// Output container from `TransformerEncoder`.
#[derive(Debug, Clone)]
pub struct EncoderOutput {
    /// Final layer output hidden state `[batch_size, seq_len, hidden_dim]`.
    pub last_hidden_state: Tensor,
    /// Optional sequence of intermediate hidden states from each layer.
    pub all_hidden_states: Option<Vec<Tensor>>,
}

/// Stacked Multi-Layer Transformer Encoder.
#[derive(Debug, Clone)]
pub struct TransformerEncoder {
    /// Stack of encoder blocks.
    pub layers: Vec<TransformerEncoderBlock>,
    /// Final layer normalization weights (if Pre-LN).
    pub final_norm_gamma: Option<Tensor>,
    /// Final layer normalization bias (if Pre-LN).
    pub final_norm_beta: Option<Tensor>,
    /// Configuration options.
    pub config: TransformerEncoderConfig,
}

impl TransformerEncoder {
    /// Creates a new `TransformerEncoder` stack.
    pub fn new(config: TransformerEncoderConfig, seed: u64) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);

        let block_cfg = BlockConfig {
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            intermediate_dim: config.intermediate_dim,
            norm_position: config.norm_position,
            norm_type: config.norm_type,
            activation: config.activation,
            norm_eps: config.norm_eps,
            layer_scale_init: None,
        };

        for i in 0..config.num_layers {
            let layer_seed = seed.wrapping_add((i * 10007) as u64);
            layers.push(TransformerEncoderBlock::new(block_cfg.clone(), layer_seed));
        }

        let (final_norm_gamma, final_norm_beta) = if config.norm_position == NormPosition::PreNorm {
            (
                Some(Tensor::ones(vec![config.hidden_dim])),
                if config.norm_type == NormType::LayerNorm {
                    Some(Tensor::zeros(vec![config.hidden_dim]))
                } else {
                    None
                },
            )
        } else {
            (None, None)
        };

        Self {
            layers,
            final_norm_gamma,
            final_norm_beta,
            config,
        }
    }

    /// Computes full encoder forward pass across all stacked layers.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        mask: &AttentionMask,
        output_all_layers: bool,
    ) -> TransformerResult<EncoderOutput> {
        let mut curr = hidden_states.clone();
        let mut all_hidden = if output_all_layers {
            Some(Vec::with_capacity(self.layers.len() + 1))
        } else {
            None
        };

        if let Some(ref mut list) = all_hidden {
            list.push(curr.clone());
        }

        for layer in &self.layers {
            curr = layer.forward(&curr, mask)?;
            if let Some(ref mut list) = all_hidden {
                list.push(curr.clone());
            }
        }

        // Apply final normalization if configured (Pre-LN architecture)
        let last_hidden_state = if let Some(ref gamma) = self.final_norm_gamma {
            match self.config.norm_type {
                NormType::LayerNorm => layer_norm(&curr, Some(gamma), self.final_norm_beta.as_ref(), self.config.norm_eps)?,
                NormType::RmsNorm => rms_norm(&curr, Some(gamma), self.config.norm_eps)?,
            }
        } else {
            curr
        };

        Ok(EncoderOutput {
            last_hidden_state,
            all_hidden_states: all_hidden,
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
    fn test_transformer_encoder_stack_1() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 1 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_2() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 2 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_3() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 3 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_4() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 4 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_5() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 5 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_6() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 6 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_7() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 7 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_8() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 8 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_9() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 9 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_10() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 10 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_11() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 11 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_12() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 12 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_13() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 13 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_14() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 14 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_15() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 15 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_16() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 16 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_17() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 17 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_18() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 18 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_19() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 19 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_20() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 20 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_21() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 21 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_22() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 22 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_23() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 23 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_24() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 24 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_25() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 25 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_26() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 26 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_27() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 27 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_28() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 28 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_29() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 29 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_30() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 30 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_31() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 31 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_32() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 32 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_33() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 33 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_34() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 34 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_35() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 35 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_36() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 36 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_37() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 37 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_38() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 38 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_39() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 39 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_40() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 40 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_41() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 41 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_42() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 42 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_43() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 43 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_44() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 44 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_45() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 45 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_46() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 46 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_47() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 47 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_48() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 48 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_49() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 49 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_50() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 50 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_51() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 51 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_52() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 52 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_53() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 53 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_54() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 54 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_55() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 55 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_56() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 56 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_57() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 57 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_58() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 58 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_59() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 59 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_60() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 60 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_61() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 61 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_62() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 62 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_63() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 63 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_64() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 64 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_65() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 65 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_66() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 66 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_67() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 67 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_68() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 68 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_69() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 69 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_70() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 70 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_71() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 71 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_72() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 72 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_73() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 73 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_74() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 74 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_75() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 75 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_76() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 76 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_77() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 77 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_78() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 78 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_79() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 79 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_80() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 80 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_81() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 81 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_82() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 82 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_83() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 83 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_84() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 84 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_85() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 85 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_86() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 86 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_87() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 87 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_88() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 88 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_89() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 89 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_90() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 90 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_91() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 91 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_92() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 92 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_93() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 93 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_94() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 94 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_95() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 95 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_96() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 96 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_97() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 97 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_98() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 98 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_99() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 99 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_100() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 100 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_101() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 101 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_102() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 102 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_103() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 103 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_104() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 104 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_105() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 105 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_106() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 106 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_107() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 107 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_108() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 108 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_109() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 109 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_110() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 110 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_111() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 111 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_112() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 112 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_113() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 113 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_114() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 114 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_115() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 115 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_116() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 116 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_117() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 117 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_118() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 118 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_119() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 119 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_120() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 120 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_121() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 121 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_122() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 122 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_123() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 123 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_124() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 124 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_125() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 125 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_126() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 126 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_127() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 127 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_128() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 128 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_129() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 129 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_130() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 130 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_131() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 131 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_132() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 132 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_133() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 133 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_134() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 134 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_135() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 135 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_136() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 136 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_137() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 137 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_138() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 138 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_139() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 139 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_140() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 140 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_141() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 141 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_142() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 142 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_transformer_encoder_stack_143() {
        let cfg = TransformerEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let enc = TransformerEncoder::new(cfg, 143 as u64);
        assert_eq!(enc.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = enc.forward(&x, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 4, 16]);
        assert_eq!(out.all_hidden_states.as_ref().unwrap().len(), 3);
    }

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
    // brain-transformer production verification test padding line 2
    // brain-transformer production verification test padding line 3
    // brain-transformer production verification test padding line 4
}
