//! # Transformer Encoder Stack
//!
//! Stacked Transformer Encoder layers with hidden-state extraction, pre/post normalization, and bidirectional representation output.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

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
                NormType::LayerNorm => layer_norm(
                    &curr,
                    Some(gamma),
                    self.final_norm_beta.as_ref(),
                    self.config.norm_eps,
                )?,
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
}
