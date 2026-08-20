//! # Transformer Encoder Block
//!
//! Unified self-attention and feed-forward sub-layer with Pre-LN / Post-LN normalization placement and residual skip connections.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::attention::multi_head::{MhaConfig, MultiHeadAttention};
use crate::config::{ActivationType, FfnConfig, NormPosition, NormType};
use crate::core::{AttentionMask, TransformerResult};
use crate::ffn::FeedForwardNetwork;
use crate::ops::{layer_norm, rms_norm};
use brain_core::Tensor;

/// Configuration for individual encoder block.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockConfig {
    /// Hidden representation dimension.
    pub hidden_dim: usize,
    /// Number of self-attention heads.
    pub num_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Intermediate expansion dimension for FFN.
    pub intermediate_dim: usize,
    /// Normalization placement strategy.
    pub norm_position: NormPosition,
    /// Normalization algorithm variant.
    pub norm_type: NormType,
    /// Non-linear activation.
    pub activation: ActivationType,
    /// Normalization epsilon.
    pub norm_eps: f64,
    /// LayerScale initial scaling factor (optional, e.g. 1e-5).
    pub layer_scale_init: Option<f64>,
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            num_heads: 12,
            head_dim: 64,
            intermediate_dim: 3072,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-6,
            layer_scale_init: None,
        }
    }
}

/// Production Transformer Encoder Block.
#[derive(Debug, Clone)]
pub struct TransformerEncoderBlock {
    /// Self-attention sub-layer.
    pub self_attn: MultiHeadAttention,
    /// Feed-forward sub-layer.
    pub ffn: FeedForwardNetwork,
    /// Pre/Post attention normalization weights.
    pub norm1_gamma: Tensor,
    pub norm1_beta: Option<Tensor>,
    /// Pre/Post FFN normalization weights.
    pub norm2_gamma: Tensor,
    pub norm2_beta: Option<Tensor>,
    /// Optional LayerScale parameter vector for attention.
    pub layer_scale1: Option<Tensor>,
    /// Optional LayerScale parameter vector for FFN.
    pub layer_scale2: Option<Tensor>,
    /// Configuration options.
    pub config: BlockConfig,
}

impl TransformerEncoderBlock {
    /// Creates a new `TransformerEncoderBlock` with initialized weights.
    pub fn new(config: BlockConfig, seed: u64) -> Self {
        let mha_cfg = MhaConfig {
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            dropout: 0.0,
            bias: false,
            is_causal: false,
        };
        let self_attn = MultiHeadAttention::new(mha_cfg, seed);

        let ffn_cfg = FfnConfig {
            hidden_dim: config.hidden_dim,
            intermediate_dim: config.intermediate_dim,
            activation: config.activation,
            dropout: 0.0,
            bias: false,
        };
        let ffn = FeedForwardNetwork::new(ffn_cfg, seed.wrapping_add(500));

        let norm1_gamma = Tensor::ones(vec![config.hidden_dim]);
        let norm1_beta = if config.norm_type == NormType::LayerNorm {
            Some(Tensor::zeros(vec![config.hidden_dim]))
        } else {
            None
        };

        let norm2_gamma = Tensor::ones(vec![config.hidden_dim]);
        let norm2_beta = if config.norm_type == NormType::LayerNorm {
            Some(Tensor::zeros(vec![config.hidden_dim]))
        } else {
            None
        };

        let (layer_scale1, layer_scale2) = if let Some(init_val) = config.layer_scale_init {
            (
                Some(Tensor::from_vec(
                    vec![init_val; config.hidden_dim],
                    vec![config.hidden_dim],
                )),
                Some(Tensor::from_vec(
                    vec![init_val; config.hidden_dim],
                    vec![config.hidden_dim],
                )),
            )
        } else {
            (None, None)
        };

        Self {
            self_attn,
            ffn,
            norm1_gamma,
            norm1_beta,
            norm2_gamma,
            norm2_beta,
            layer_scale1,
            layer_scale2,
            config,
        }
    }

    fn apply_norm(
        &self,
        x: &Tensor,
        gamma: &Tensor,
        beta: Option<&Tensor>,
    ) -> TransformerResult<Tensor> {
        match self.config.norm_type {
            NormType::LayerNorm => layer_norm(x, Some(gamma), beta, self.config.norm_eps),
            NormType::RmsNorm => rms_norm(x, Some(gamma), self.config.norm_eps),
        }
    }

    /// Executes encoder block forward pass with residual connections.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        mask: &AttentionMask,
    ) -> TransformerResult<Tensor> {
        match self.config.norm_position {
            NormPosition::PreNorm => {
                // 1. Self-Attention with Pre-LN
                let norm_x1 =
                    self.apply_norm(hidden_states, &self.norm1_gamma, self.norm1_beta.as_ref())?;
                let attn_out = self.self_attn.forward_mha(&norm_x1, None, mask)?;

                // Residual 1
                let mut h1_data = hidden_states.data().to_vec();
                let attn_data = attn_out.data();
                for i in 0..h1_data.len() {
                    h1_data[i] += attn_data[i];
                }
                let h1 = Tensor::from_vec(h1_data, hidden_states.shape().to_vec());

                // 2. FFN with Pre-LN
                let norm_x2 = self.apply_norm(&h1, &self.norm2_gamma, self.norm2_beta.as_ref())?;
                let ffn_out = self.ffn.forward(&norm_x2)?;

                // Residual 2
                let mut h2_data = h1.data().to_vec();
                let ffn_data = ffn_out.data();
                for i in 0..h2_data.len() {
                    h2_data[i] += ffn_data[i];
                }
                Ok(Tensor::from_vec(h2_data, hidden_states.shape().to_vec()))
            }
            NormPosition::PostNorm => {
                // 1. Self-Attention with Post-LN
                let attn_out = self.self_attn.forward_mha(hidden_states, None, mask)?;
                let mut h1_data = hidden_states.data().to_vec();
                let attn_data = attn_out.data();
                for i in 0..h1_data.len() {
                    h1_data[i] += attn_data[i];
                }
                let h1_res = Tensor::from_vec(h1_data, hidden_states.shape().to_vec());
                let h1 = self.apply_norm(&h1_res, &self.norm1_gamma, self.norm1_beta.as_ref())?;

                // 2. FFN with Post-LN
                let ffn_out = self.ffn.forward(&h1)?;
                let mut h2_data = h1.data().to_vec();
                let ffn_data = ffn_out.data();
                for i in 0..h2_data.len() {
                    h2_data[i] += ffn_data[i];
                }
                let h2_res = Tensor::from_vec(h2_data, hidden_states.shape().to_vec());
                self.apply_norm(&h2_res, &self.norm2_gamma, self.norm2_beta.as_ref())
            }
            NormPosition::SandwichNorm => {
                // Pre-norm then Post-norm
                let norm_pre1 =
                    self.apply_norm(hidden_states, &self.norm1_gamma, self.norm1_beta.as_ref())?;
                let attn_out = self.self_attn.forward_mha(&norm_pre1, None, mask)?;
                let mut h1_data = hidden_states.data().to_vec();
                let attn_data = attn_out.data();
                for i in 0..h1_data.len() {
                    h1_data[i] += attn_data[i];
                }
                let h1_res = Tensor::from_vec(h1_data, hidden_states.shape().to_vec());
                let h1 = self.apply_norm(&h1_res, &self.norm1_gamma, self.norm1_beta.as_ref())?;

                let norm_pre2 =
                    self.apply_norm(&h1, &self.norm2_gamma, self.norm2_beta.as_ref())?;
                let ffn_out = self.ffn.forward(&norm_pre2)?;
                let mut h2_data = h1.data().to_vec();
                let ffn_data = ffn_out.data();
                for i in 0..h2_data.len() {
                    h2_data[i] += ffn_data[i];
                }
                let h2_res = Tensor::from_vec(h2_data, hidden_states.shape().to_vec());
                self.apply_norm(&h2_res, &self.norm2_gamma, self.norm2_beta.as_ref())
            }
        }
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
    fn test_encoder_block_1() {
        let cfg = BlockConfig {
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::LayerNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
            layer_scale_init: None,
        };
        let block = TransformerEncoderBlock::new(cfg, 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }
}
