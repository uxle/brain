//! # Transformer Decoder Layer
//!
//! Masked causal self-attention + optional encoder-decoder cross-attention + Feed-Forward Network sub-layers.
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
use crate::decoder::cross::{CrossAttention, CrossAttnConfig};
use crate::ffn::FeedForwardNetwork;
use crate::ops::{layer_norm, rms_norm};
use brain_core::Tensor;

/// Configuration for individual decoder layer.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderLayerConfig {
    /// Hidden representation dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Intermediate expansion dimension for FFN.
    pub intermediate_dim: usize,
    /// Whether this decoder includes encoder-decoder cross-attention (true for Seq2Seq, false for Causal LM).
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

impl Default for DecoderLayerConfig {
    fn default() -> Self {
        Self {
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

/// Production Transformer Decoder Layer.
#[derive(Debug, Clone)]
pub struct TransformerDecoderLayer {
    /// Causal self-attention sub-layer.
    pub self_attn: MultiHeadAttention,
    /// Optional encoder-decoder cross-attention sub-layer.
    pub cross_attn: Option<CrossAttention>,
    /// Feed-Forward sub-layer.
    pub ffn: FeedForwardNetwork,
    /// Self-attention norm parameters.
    pub norm1_gamma: Tensor,
    pub norm1_beta: Option<Tensor>,
    /// Cross-attention norm parameters.
    pub norm2_gamma: Option<Tensor>,
    pub norm2_beta: Option<Tensor>,
    /// FFN norm parameters.
    pub norm3_gamma: Tensor,
    pub norm3_beta: Option<Tensor>,
    /// Configuration options.
    pub config: DecoderLayerConfig,
}

impl TransformerDecoderLayer {
    /// Creates a new `TransformerDecoderLayer`.
    pub fn new(config: DecoderLayerConfig, seed: u64) -> Self {
        let mha_cfg = MhaConfig {
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            dropout: 0.0,
            bias: false,
            is_causal: true,
        };
        let self_attn = MultiHeadAttention::new(mha_cfg, seed);

        let cross_attn = if config.has_cross_attention {
            let cross_cfg = CrossAttnConfig {
                hidden_dim: config.hidden_dim,
                num_heads: config.num_heads,
                head_dim: config.head_dim,
                bias: false,
                dropout: 0.0,
            };
            Some(CrossAttention::new(cross_cfg, seed.wrapping_add(200)))
        } else {
            None
        };

        let ffn_cfg = FfnConfig {
            hidden_dim: config.hidden_dim,
            intermediate_dim: config.intermediate_dim,
            activation: config.activation,
            dropout: 0.0,
            bias: false,
        };
        let ffn = FeedForwardNetwork::new(ffn_cfg, seed.wrapping_add(400));

        let norm1_gamma = Tensor::ones(vec![config.hidden_dim]);
        let norm1_beta = if config.norm_type == NormType::LayerNorm {
            Some(Tensor::zeros(vec![config.hidden_dim]))
        } else {
            None
        };

        let (norm2_gamma, norm2_beta) = if config.has_cross_attention {
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

        let norm3_gamma = Tensor::ones(vec![config.hidden_dim]);
        let norm3_beta = if config.norm_type == NormType::LayerNorm {
            Some(Tensor::zeros(vec![config.hidden_dim]))
        } else {
            None
        };

        Self {
            self_attn,
            cross_attn,
            ffn,
            norm1_gamma,
            norm1_beta,
            norm2_gamma,
            norm2_beta,
            norm3_gamma,
            norm3_beta,
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

    /// Executes decoder layer forward pass.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        self_mask: &AttentionMask,
        cross_mask: &AttentionMask,
    ) -> TransformerResult<Tensor> {
        // 1. Causal Self-Attention
        let norm_x1 =
            self.apply_norm(hidden_states, &self.norm1_gamma, self.norm1_beta.as_ref())?;
        let self_out = self.self_attn.forward_mha(&norm_x1, None, self_mask)?;

        let mut h1_data = hidden_states.data().to_vec();
        let self_data = self_out.data();
        for i in 0..h1_data.len() {
            h1_data[i] += self_data[i];
        }
        let mut curr = Tensor::from_vec(h1_data, hidden_states.shape().to_vec());

        // 2. Optional Cross-Attention
        if let (Some(ref cross), Some(enc_states), Some(ref gamma2)) =
            (&self.cross_attn, encoder_hidden_states, &self.norm2_gamma)
        {
            let norm_x2 = self.apply_norm(&curr, gamma2, self.norm2_beta.as_ref())?;
            let cross_out = cross.forward(&norm_x2, enc_states, cross_mask)?;

            let mut h2_data = curr.data().to_vec();
            let cross_data = cross_out.data();
            for i in 0..h2_data.len() {
                h2_data[i] += cross_data[i];
            }
            curr = Tensor::from_vec(h2_data, curr.shape().to_vec());
        }

        // 3. FFN
        let norm_x3 = self.apply_norm(&curr, &self.norm3_gamma, self.norm3_beta.as_ref())?;
        let ffn_out = self.ffn.forward(&norm_x3)?;

        let mut h3_data = curr.data().to_vec();
        let ffn_data = ffn_out.data();
        for i in 0..h3_data.len() {
            h3_data[i] += ffn_data[i];
        }

        Ok(Tensor::from_vec(h3_data, curr.shape().to_vec()))
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
    fn test_decoder_layer_1() {
        let cfg = DecoderLayerConfig {
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: true,
            ..Default::default()
        };
        let layer = TransformerDecoderLayer::new(cfg, 1 as u64);
        let dec_x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let enc_x = Tensor::from_vec(vec![2.0; 2 * 4 * 16], vec![2, 4, 16]);

        let out = layer
            .forward(
                &dec_x,
                Some(&enc_x),
                &AttentionMask::Causal,
                &AttentionMask::None,
            )
            .unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }
}
