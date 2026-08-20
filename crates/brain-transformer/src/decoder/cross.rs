//! # Cross-Attention (Encoder-Decoder Attention)
//!
//! Multi-head cross-attention projecting queries from decoder states and keys/values from encoder memory representations.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::attention::scaled::scaled_dot_product_attention;
use crate::core::{AttentionMask, LinearParams, TransformerError, TransformerResult};
use brain_core::Tensor;

/// Configuration for cross-attention sub-layer.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossAttnConfig {
    /// Hidden representation dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Bias flag.
    pub bias: bool,
    /// Attention dropout rate.
    pub dropout: f32,
}

impl Default for CrossAttnConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            num_heads: 12,
            head_dim: 64,
            bias: false,
            dropout: 0.0,
        }
    }
}

/// Cross-Attention Layer bridging Decoder queries to Encoder keys/values.
#[derive(Debug, Clone)]
pub struct CrossAttention {
    /// Query projection from decoder state.
    pub q_proj: LinearParams,
    /// Key projection from encoder output representation.
    pub k_proj: LinearParams,
    /// Value projection from encoder output representation.
    pub v_proj: LinearParams,
    /// Output projection.
    pub out_proj: LinearParams,
    /// Configuration options.
    pub config: CrossAttnConfig,
}

impl CrossAttention {
    /// Creates a new `CrossAttention` layer.
    pub fn new(config: CrossAttnConfig, seed: u64) -> Self {
        let q_proj = LinearParams::new(config.hidden_dim, config.hidden_dim, config.bias, seed);
        let k_proj = LinearParams::new(
            config.hidden_dim,
            config.hidden_dim,
            config.bias,
            seed.wrapping_add(100),
        );
        let v_proj = LinearParams::new(
            config.hidden_dim,
            config.hidden_dim,
            config.bias,
            seed.wrapping_add(200),
        );
        let out_proj = LinearParams::new(
            config.hidden_dim,
            config.hidden_dim,
            config.bias,
            seed.wrapping_add(300),
        );

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            config,
        }
    }

    /// Executes cross-attention forward pass.
    pub fn forward(
        &self,
        decoder_hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        mask: &AttentionMask,
    ) -> TransformerResult<Tensor> {
        let q_shape = decoder_hidden_states.shape();
        let k_shape = encoder_hidden_states.shape();

        let batch_size = q_shape[0];
        let seq_q = q_shape[1];
        let seq_k = k_shape[1];

        let q = self.q_proj.forward(decoder_hidden_states)?;
        let k = self.k_proj.forward(encoder_hidden_states)?;
        let v = self.v_proj.forward(encoder_hidden_states)?;

        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        let q_4d = Tensor::from_vec(
            q.data().to_vec(),
            vec![batch_size, num_heads, seq_q, head_dim],
        );
        let k_4d = Tensor::from_vec(
            k.data().to_vec(),
            vec![batch_size, num_heads, seq_k, head_dim],
        );
        let v_4d = Tensor::from_vec(
            v.data().to_vec(),
            vec![batch_size, num_heads, seq_k, head_dim],
        );

        let (attn_out, _) = scaled_dot_product_attention(&q_4d, &k_4d, &v_4d, mask, None)?;
        let merged = Tensor::from_vec(
            attn_out.data().to_vec(),
            vec![batch_size, seq_q, num_heads * head_dim],
        );

        self.out_proj.forward(&merged)
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
    fn test_cross_attention_1() {
        let cfg = CrossAttnConfig {
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            bias: false,
            dropout: 0.0,
        };
        let cross = CrossAttention::new(cfg, 1 as u64);
        let dec_x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let enc_x = Tensor::from_vec(vec![2.0; 2 * 5 * 16], vec![2, 5, 16]);

        let out = cross.forward(&dec_x, &enc_x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }
}
