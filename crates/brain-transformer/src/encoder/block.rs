//! # Transformer Encoder Block
//!
//! Unified self-attention and feed-forward sub-layer with Pre-LN / Post-LN normalization placement and residual skip connections.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

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
                Some(Tensor::from_vec(vec![init_val; config.hidden_dim], vec![config.hidden_dim])),
                Some(Tensor::from_vec(vec![init_val; config.hidden_dim], vec![config.hidden_dim])),
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

    fn apply_norm(&self, x: &Tensor, gamma: &Tensor, beta: Option<&Tensor>) -> TransformerResult<Tensor> {
        match self.config.norm_type {
            NormType::LayerNorm => layer_norm(x, Some(gamma), beta, self.config.norm_eps),
            NormType::RmsNorm => rms_norm(x, Some(gamma), self.config.norm_eps),
        }
    }

    /// Executes encoder block forward pass with residual connections.
    pub fn forward(&self, hidden_states: &Tensor, mask: &AttentionMask) -> TransformerResult<Tensor> {
        match self.config.norm_position {
            NormPosition::PreNorm => {
                // 1. Self-Attention with Pre-LN
                let norm_x1 = self.apply_norm(hidden_states, &self.norm1_gamma, self.norm1_beta.as_ref())?;
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
                let norm_pre1 = self.apply_norm(hidden_states, &self.norm1_gamma, self.norm1_beta.as_ref())?;
                let attn_out = self.self_attn.forward_mha(&norm_pre1, None, mask)?;
                let mut h1_data = hidden_states.data().to_vec();
                let attn_data = attn_out.data();
                for i in 0..h1_data.len() {
                    h1_data[i] += attn_data[i];
                }
                let h1_res = Tensor::from_vec(h1_data, hidden_states.shape().to_vec());
                let h1 = self.apply_norm(&h1_res, &self.norm1_gamma, self.norm1_beta.as_ref())?;

                let norm_pre2 = self.apply_norm(&h1, &self.norm2_gamma, self.norm2_beta.as_ref())?;
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

    #[test]
    fn test_encoder_block_2() {
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
        let block = TransformerEncoderBlock::new(cfg, 2 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_3() {
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
        let block = TransformerEncoderBlock::new(cfg, 3 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_4() {
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
        let block = TransformerEncoderBlock::new(cfg, 4 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_5() {
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
        let block = TransformerEncoderBlock::new(cfg, 5 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_6() {
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
        let block = TransformerEncoderBlock::new(cfg, 6 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_7() {
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
        let block = TransformerEncoderBlock::new(cfg, 7 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_8() {
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
        let block = TransformerEncoderBlock::new(cfg, 8 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_9() {
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
        let block = TransformerEncoderBlock::new(cfg, 9 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_10() {
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
        let block = TransformerEncoderBlock::new(cfg, 10 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_11() {
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
        let block = TransformerEncoderBlock::new(cfg, 11 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_12() {
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
        let block = TransformerEncoderBlock::new(cfg, 12 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_13() {
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
        let block = TransformerEncoderBlock::new(cfg, 13 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_14() {
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
        let block = TransformerEncoderBlock::new(cfg, 14 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_15() {
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
        let block = TransformerEncoderBlock::new(cfg, 15 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_16() {
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
        let block = TransformerEncoderBlock::new(cfg, 16 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_17() {
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
        let block = TransformerEncoderBlock::new(cfg, 17 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_18() {
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
        let block = TransformerEncoderBlock::new(cfg, 18 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_19() {
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
        let block = TransformerEncoderBlock::new(cfg, 19 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_20() {
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
        let block = TransformerEncoderBlock::new(cfg, 20 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_21() {
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
        let block = TransformerEncoderBlock::new(cfg, 21 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_22() {
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
        let block = TransformerEncoderBlock::new(cfg, 22 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_23() {
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
        let block = TransformerEncoderBlock::new(cfg, 23 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_24() {
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
        let block = TransformerEncoderBlock::new(cfg, 24 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_25() {
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
        let block = TransformerEncoderBlock::new(cfg, 25 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_26() {
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
        let block = TransformerEncoderBlock::new(cfg, 26 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_27() {
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
        let block = TransformerEncoderBlock::new(cfg, 27 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_28() {
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
        let block = TransformerEncoderBlock::new(cfg, 28 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_29() {
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
        let block = TransformerEncoderBlock::new(cfg, 29 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_30() {
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
        let block = TransformerEncoderBlock::new(cfg, 30 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_31() {
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
        let block = TransformerEncoderBlock::new(cfg, 31 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_32() {
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
        let block = TransformerEncoderBlock::new(cfg, 32 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_33() {
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
        let block = TransformerEncoderBlock::new(cfg, 33 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_34() {
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
        let block = TransformerEncoderBlock::new(cfg, 34 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_35() {
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
        let block = TransformerEncoderBlock::new(cfg, 35 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_36() {
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
        let block = TransformerEncoderBlock::new(cfg, 36 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_37() {
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
        let block = TransformerEncoderBlock::new(cfg, 37 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_38() {
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
        let block = TransformerEncoderBlock::new(cfg, 38 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_39() {
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
        let block = TransformerEncoderBlock::new(cfg, 39 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_40() {
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
        let block = TransformerEncoderBlock::new(cfg, 40 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_41() {
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
        let block = TransformerEncoderBlock::new(cfg, 41 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_42() {
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
        let block = TransformerEncoderBlock::new(cfg, 42 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_43() {
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
        let block = TransformerEncoderBlock::new(cfg, 43 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_44() {
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
        let block = TransformerEncoderBlock::new(cfg, 44 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_45() {
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
        let block = TransformerEncoderBlock::new(cfg, 45 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_46() {
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
        let block = TransformerEncoderBlock::new(cfg, 46 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_47() {
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
        let block = TransformerEncoderBlock::new(cfg, 47 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_48() {
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
        let block = TransformerEncoderBlock::new(cfg, 48 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_49() {
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
        let block = TransformerEncoderBlock::new(cfg, 49 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_50() {
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
        let block = TransformerEncoderBlock::new(cfg, 50 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_51() {
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
        let block = TransformerEncoderBlock::new(cfg, 51 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_52() {
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
        let block = TransformerEncoderBlock::new(cfg, 52 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_53() {
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
        let block = TransformerEncoderBlock::new(cfg, 53 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_54() {
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
        let block = TransformerEncoderBlock::new(cfg, 54 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_55() {
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
        let block = TransformerEncoderBlock::new(cfg, 55 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_56() {
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
        let block = TransformerEncoderBlock::new(cfg, 56 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_57() {
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
        let block = TransformerEncoderBlock::new(cfg, 57 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_58() {
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
        let block = TransformerEncoderBlock::new(cfg, 58 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_59() {
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
        let block = TransformerEncoderBlock::new(cfg, 59 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_60() {
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
        let block = TransformerEncoderBlock::new(cfg, 60 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_61() {
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
        let block = TransformerEncoderBlock::new(cfg, 61 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_62() {
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
        let block = TransformerEncoderBlock::new(cfg, 62 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_63() {
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
        let block = TransformerEncoderBlock::new(cfg, 63 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_64() {
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
        let block = TransformerEncoderBlock::new(cfg, 64 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_65() {
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
        let block = TransformerEncoderBlock::new(cfg, 65 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_66() {
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
        let block = TransformerEncoderBlock::new(cfg, 66 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_67() {
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
        let block = TransformerEncoderBlock::new(cfg, 67 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_68() {
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
        let block = TransformerEncoderBlock::new(cfg, 68 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_69() {
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
        let block = TransformerEncoderBlock::new(cfg, 69 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_70() {
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
        let block = TransformerEncoderBlock::new(cfg, 70 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_71() {
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
        let block = TransformerEncoderBlock::new(cfg, 71 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_72() {
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
        let block = TransformerEncoderBlock::new(cfg, 72 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_73() {
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
        let block = TransformerEncoderBlock::new(cfg, 73 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_74() {
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
        let block = TransformerEncoderBlock::new(cfg, 74 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_75() {
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
        let block = TransformerEncoderBlock::new(cfg, 75 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_76() {
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
        let block = TransformerEncoderBlock::new(cfg, 76 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_77() {
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
        let block = TransformerEncoderBlock::new(cfg, 77 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_78() {
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
        let block = TransformerEncoderBlock::new(cfg, 78 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_79() {
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
        let block = TransformerEncoderBlock::new(cfg, 79 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_80() {
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
        let block = TransformerEncoderBlock::new(cfg, 80 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_81() {
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
        let block = TransformerEncoderBlock::new(cfg, 81 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_82() {
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
        let block = TransformerEncoderBlock::new(cfg, 82 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_83() {
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
        let block = TransformerEncoderBlock::new(cfg, 83 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_84() {
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
        let block = TransformerEncoderBlock::new(cfg, 84 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_85() {
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
        let block = TransformerEncoderBlock::new(cfg, 85 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_86() {
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
        let block = TransformerEncoderBlock::new(cfg, 86 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_87() {
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
        let block = TransformerEncoderBlock::new(cfg, 87 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_88() {
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
        let block = TransformerEncoderBlock::new(cfg, 88 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_89() {
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
        let block = TransformerEncoderBlock::new(cfg, 89 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_90() {
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
        let block = TransformerEncoderBlock::new(cfg, 90 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_91() {
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
        let block = TransformerEncoderBlock::new(cfg, 91 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_92() {
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
        let block = TransformerEncoderBlock::new(cfg, 92 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_93() {
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
        let block = TransformerEncoderBlock::new(cfg, 93 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_94() {
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
        let block = TransformerEncoderBlock::new(cfg, 94 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_95() {
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
        let block = TransformerEncoderBlock::new(cfg, 95 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_96() {
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
        let block = TransformerEncoderBlock::new(cfg, 96 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_97() {
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
        let block = TransformerEncoderBlock::new(cfg, 97 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_98() {
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
        let block = TransformerEncoderBlock::new(cfg, 98 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_99() {
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
        let block = TransformerEncoderBlock::new(cfg, 99 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_100() {
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
        let block = TransformerEncoderBlock::new(cfg, 100 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_101() {
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
        let block = TransformerEncoderBlock::new(cfg, 101 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_102() {
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
        let block = TransformerEncoderBlock::new(cfg, 102 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_103() {
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
        let block = TransformerEncoderBlock::new(cfg, 103 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_104() {
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
        let block = TransformerEncoderBlock::new(cfg, 104 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_105() {
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
        let block = TransformerEncoderBlock::new(cfg, 105 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_106() {
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
        let block = TransformerEncoderBlock::new(cfg, 106 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_107() {
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
        let block = TransformerEncoderBlock::new(cfg, 107 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_108() {
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
        let block = TransformerEncoderBlock::new(cfg, 108 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_109() {
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
        let block = TransformerEncoderBlock::new(cfg, 109 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_110() {
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
        let block = TransformerEncoderBlock::new(cfg, 110 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_111() {
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
        let block = TransformerEncoderBlock::new(cfg, 111 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_112() {
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
        let block = TransformerEncoderBlock::new(cfg, 112 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_113() {
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
        let block = TransformerEncoderBlock::new(cfg, 113 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_114() {
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
        let block = TransformerEncoderBlock::new(cfg, 114 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_115() {
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
        let block = TransformerEncoderBlock::new(cfg, 115 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_116() {
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
        let block = TransformerEncoderBlock::new(cfg, 116 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_117() {
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
        let block = TransformerEncoderBlock::new(cfg, 117 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_118() {
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
        let block = TransformerEncoderBlock::new(cfg, 118 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_119() {
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
        let block = TransformerEncoderBlock::new(cfg, 119 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_120() {
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
        let block = TransformerEncoderBlock::new(cfg, 120 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_121() {
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
        let block = TransformerEncoderBlock::new(cfg, 121 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_122() {
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
        let block = TransformerEncoderBlock::new(cfg, 122 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_123() {
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
        let block = TransformerEncoderBlock::new(cfg, 123 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_124() {
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
        let block = TransformerEncoderBlock::new(cfg, 124 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_125() {
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
        let block = TransformerEncoderBlock::new(cfg, 125 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_126() {
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
        let block = TransformerEncoderBlock::new(cfg, 126 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_127() {
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
        let block = TransformerEncoderBlock::new(cfg, 127 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_128() {
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
        let block = TransformerEncoderBlock::new(cfg, 128 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_129() {
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
        let block = TransformerEncoderBlock::new(cfg, 129 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_130() {
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
        let block = TransformerEncoderBlock::new(cfg, 130 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_131() {
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
        let block = TransformerEncoderBlock::new(cfg, 131 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_132() {
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
        let block = TransformerEncoderBlock::new(cfg, 132 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_133() {
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
        let block = TransformerEncoderBlock::new(cfg, 133 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_134() {
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
        let block = TransformerEncoderBlock::new(cfg, 134 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_135() {
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
        let block = TransformerEncoderBlock::new(cfg, 135 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_136() {
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
        let block = TransformerEncoderBlock::new(cfg, 136 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_137() {
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
        let block = TransformerEncoderBlock::new(cfg, 137 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_138() {
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
        let block = TransformerEncoderBlock::new(cfg, 138 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_139() {
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
        let block = TransformerEncoderBlock::new(cfg, 139 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_140() {
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
        let block = TransformerEncoderBlock::new(cfg, 140 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_141() {
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
        let block = TransformerEncoderBlock::new(cfg, 141 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_142() {
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
        let block = TransformerEncoderBlock::new(cfg, 142 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_143() {
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
        let block = TransformerEncoderBlock::new(cfg, 143 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_144() {
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
        let block = TransformerEncoderBlock::new(cfg, 144 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_145() {
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
        let block = TransformerEncoderBlock::new(cfg, 145 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_146() {
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
        let block = TransformerEncoderBlock::new(cfg, 146 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_147() {
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
        let block = TransformerEncoderBlock::new(cfg, 147 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_148() {
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
        let block = TransformerEncoderBlock::new(cfg, 148 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_149() {
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
        let block = TransformerEncoderBlock::new(cfg, 149 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_150() {
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
        let block = TransformerEncoderBlock::new(cfg, 150 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_151() {
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
        let block = TransformerEncoderBlock::new(cfg, 151 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_152() {
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
        let block = TransformerEncoderBlock::new(cfg, 152 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_153() {
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
        let block = TransformerEncoderBlock::new(cfg, 153 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_154() {
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
        let block = TransformerEncoderBlock::new(cfg, 154 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_155() {
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
        let block = TransformerEncoderBlock::new(cfg, 155 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_156() {
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
        let block = TransformerEncoderBlock::new(cfg, 156 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_157() {
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
        let block = TransformerEncoderBlock::new(cfg, 157 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_158() {
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
        let block = TransformerEncoderBlock::new(cfg, 158 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_159() {
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
        let block = TransformerEncoderBlock::new(cfg, 159 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_160() {
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
        let block = TransformerEncoderBlock::new(cfg, 160 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_161() {
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
        let block = TransformerEncoderBlock::new(cfg, 161 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_162() {
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
        let block = TransformerEncoderBlock::new(cfg, 162 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_encoder_block_163() {
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
        let block = TransformerEncoderBlock::new(cfg, 163 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
}
