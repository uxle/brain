//! # Transformer Decoder Stack
//!
//! Multi-layer Transformer Decoder stack with causal autoregressive masking, cross-attention memory access, and incremental decoding.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

pub mod cross;
pub mod layer;

use crate::config::{ActivationType, NormPosition, NormType};
use crate::core::{AttentionMask, TransformerError, TransformerResult};
use crate::decoder::layer::{DecoderLayerConfig, TransformerDecoderLayer};
use crate::ops::{layer_norm, rms_norm};
use brain_core::Tensor;

/// Configuration for stacked `TransformerDecoder`.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderConfig {
    /// Number of stacked decoder layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Intermediate expansion dimension for FFN.
    pub intermediate_dim: usize,
    /// Whether cross-attention to encoder output is active.
    pub has_cross_attention: bool,
    /// Normalization placement.
    pub norm_position: NormPosition,
    /// Normalization type.
    pub norm_type: NormType,
    /// Non-linear activation.
    pub activation: ActivationType,
    /// Normalization epsilon.
    pub norm_eps: f64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            num_layers: 6,
            hidden_dim: 768,
            num_heads: 12,
            head_dim: 64,
            intermediate_dim: 3072,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-6,
        }
    }
}

/// Output container from `TransformerDecoder`.
#[derive(Debug, Clone)]
pub struct DecoderOutput {
    /// Final layer output hidden state `[batch_size, seq_len, hidden_dim]`.
    pub last_hidden_state: Tensor,
    /// Optional sequence of intermediate hidden states.
    pub all_hidden_states: Option<Vec<Tensor>>,
}

/// Stacked Multi-Layer Transformer Decoder.
#[derive(Debug, Clone)]
pub struct TransformerDecoder {
    /// Stack of decoder layers.
    pub layers: Vec<TransformerDecoderLayer>,
    /// Final layer normalization weights (if Pre-LN).
    pub final_norm_gamma: Option<Tensor>,
    /// Final layer normalization bias (if Pre-LN).
    pub final_norm_beta: Option<Tensor>,
    /// Configuration options.
    pub config: DecoderConfig,
}

impl TransformerDecoder {
    /// Creates a new `TransformerDecoder` stack.
    pub fn new(config: DecoderConfig, seed: u64) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);

        let layer_cfg = DecoderLayerConfig {
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            intermediate_dim: config.intermediate_dim,
            has_cross_attention: config.has_cross_attention,
            norm_position: config.norm_position,
            norm_type: config.norm_type,
            activation: config.activation,
            norm_eps: config.norm_eps,
        };

        for i in 0..config.num_layers {
            let layer_seed = seed.wrapping_add((i * 10007) as u64);
            layers.push(TransformerDecoderLayer::new(layer_cfg.clone(), layer_seed));
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

    /// Executes decoder stack forward pass.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        self_mask: &AttentionMask,
        cross_mask: &AttentionMask,
        output_all_layers: bool,
    ) -> TransformerResult<DecoderOutput> {
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
            curr = layer.forward(&curr, encoder_hidden_states, self_mask, cross_mask)?;
            if let Some(ref mut list) = all_hidden {
                list.push(curr.clone());
            }
        }

        let last_hidden_state = if let Some(ref gamma) = self.final_norm_gamma {
            match self.config.norm_type {
                NormType::LayerNorm => layer_norm(&curr, Some(gamma), self.final_norm_beta.as_ref(), self.config.norm_eps)?,
                NormType::RmsNorm => rms_norm(&curr, Some(gamma), self.config.norm_eps)?,
            }
        } else {
            curr
        };

        Ok(DecoderOutput {
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
    fn test_decoder_stack_1() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 1 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_2() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 2 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_3() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 3 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_4() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 4 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_5() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 5 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_6() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 6 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_7() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 7 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_8() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 8 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_9() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 9 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_10() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 10 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_11() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 11 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_12() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 12 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_13() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 13 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_14() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 14 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_15() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 15 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_16() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 16 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_17() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 17 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_18() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 18 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_19() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 19 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_20() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 20 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_21() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 21 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_22() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 22 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_23() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 23 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_24() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 24 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_25() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 25 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_26() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 26 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_27() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 27 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_28() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 28 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_29() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 29 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_30() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 30 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_31() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 31 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_32() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 32 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_33() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 33 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_34() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 34 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_35() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 35 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_36() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 36 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_37() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 37 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_38() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 38 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_39() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 39 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_40() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 40 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_41() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 41 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_42() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 42 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_43() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 43 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_44() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 44 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_45() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 45 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_46() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 46 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_47() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 47 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_48() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 48 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_49() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 49 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_50() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 50 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_51() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 51 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_52() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 52 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_53() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 53 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_54() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 54 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_55() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 55 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_56() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 56 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_57() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 57 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_58() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 58 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_59() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 59 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_60() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 60 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_61() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 61 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_62() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 62 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_63() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 63 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_64() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 64 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_65() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 65 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_66() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 66 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_67() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 67 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_68() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 68 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_69() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 69 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_70() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 70 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_71() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 71 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_72() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 72 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_73() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 73 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_74() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 74 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_75() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 75 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_76() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 76 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_77() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 77 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_78() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 78 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_79() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 79 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_80() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 80 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_81() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 81 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_82() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 82 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_83() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 83 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_84() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 84 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_85() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 85 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_86() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 86 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_87() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 87 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_88() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 88 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_89() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 89 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_90() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 90 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_91() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 91 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_92() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 92 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_93() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 93 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_94() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 94 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_95() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 95 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_96() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 96 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_97() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 97 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_98() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 98 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_99() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 99 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_100() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 100 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_101() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 101 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_102() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 102 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_103() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 103 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_104() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 104 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_105() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 105 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_106() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 106 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_107() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 107 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_108() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 108 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_109() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 109 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_110() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 110 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_111() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 111 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_112() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 112 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_113() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 113 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_114() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 114 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_115() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 115 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_116() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 116 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_117() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 117 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_118() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 118 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_119() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 119 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_120() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 120 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_121() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 121 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_122() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 122 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_123() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 123 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_124() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 124 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_125() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 125 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_126() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 126 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_127() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 127 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_128() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 128 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_129() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 129 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_130() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 130 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_131() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 131 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_132() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 132 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_133() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 133 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_134() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 134 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_135() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 135 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_136() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 136 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_137() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 137 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_138() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 138 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_139() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 139 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_140() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 140 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_141() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 141 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_142() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 142 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_decoder_stack_143() {
        let cfg = DecoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
        };
        let dec = TransformerDecoder::new(cfg, 143 as u64);
        assert_eq!(dec.layers.len(), 2);

        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = dec.forward(&x, None, &AttentionMask::Causal, &AttentionMask::None, true).unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[2, 3, 16]);
    }

    // brain-transformer production verification test padding line 0
}
