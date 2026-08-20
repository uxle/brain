//! # Feed-Forward Networks (FFN / MLP) & Gated Variants
//!
//! Standard FFN ($W_2 \text{Act}(W_1 x + b)$) and SwiGLU / GEGLU gated MLP architectures ($W_2 (\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x)$).
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::config::{ActivationType, FfnConfig};
use crate::core::{LinearParams, TransformerError, TransformerResult};
use crate::ops::{apply_activation, gelu, relu, silu};
use brain_core::Tensor;

/// Feed-Forward Network layer supporting standard 2-layer MLP and 3-layer Gated MLP (SwiGLU/GEGLU).
#[derive(Debug, Clone)]
pub struct FeedForwardNetwork {
    /// Up projection / input linear layer $W_1$ (or $W_{\text{up}}$ in gated variants).
    pub up_proj: LinearParams,
    /// Gate projection layer $W_{\text{gate}}$ (used only in SwiGLU / GEGLU).
    pub gate_proj: Option<LinearParams>,
    /// Down projection / output linear layer $W_2$.
    pub down_proj: LinearParams,
    /// Configuration options.
    pub config: FfnConfig,
}

impl FeedForwardNetwork {
    /// Creates a new `FeedForwardNetwork` with initialized parameters.
    pub fn new(config: FfnConfig, seed: u64) -> Self {
        let is_gated = matches!(
            config.activation,
            ActivationType::Swiglu | ActivationType::Geglu
        );

        let up_proj = LinearParams::new(
            config.hidden_dim,
            config.intermediate_dim,
            config.bias,
            seed,
        );
        let gate_proj = if is_gated {
            Some(LinearParams::new(
                config.hidden_dim,
                config.intermediate_dim,
                config.bias,
                seed.wrapping_add(100),
            ))
        } else {
            None
        };
        let down_proj = LinearParams::new(
            config.intermediate_dim,
            config.hidden_dim,
            config.bias,
            seed.wrapping_add(200),
        );

        Self {
            up_proj,
            gate_proj,
            down_proj,
            config,
        }
    }

    /// Computes Feed-Forward Network forward pass on representation tensor `hidden_states`.
    pub fn forward(&self, hidden_states: &Tensor) -> TransformerResult<Tensor> {
        let is_gated = matches!(
            self.config.activation,
            ActivationType::Swiglu | ActivationType::Geglu
        );

        if is_gated {
            let gate = self.gate_proj.as_ref().unwrap().forward(hidden_states)?;
            let up = self.up_proj.forward(hidden_states)?;

            let gate_data = gate.data();
            let up_data = up.data();
            let numel = gate.numel();
            let mut intermediate = vec![0.0f64; numel];

            match self.config.activation {
                ActivationType::Swiglu => {
                    for i in 0..numel {
                        intermediate[i] = silu(gate_data[i]) * up_data[i];
                    }
                }
                ActivationType::Geglu => {
                    for i in 0..numel {
                        intermediate[i] = gelu(gate_data[i]) * up_data[i];
                    }
                }
                _ => {}
            }

            let inter_tensor = Tensor::from_vec(intermediate, gate.shape().to_vec());
            self.down_proj.forward(&inter_tensor)
        } else {
            let up = self.up_proj.forward(hidden_states)?;
            let activated = apply_activation(&up, self.config.activation);
            self.down_proj.forward(&activated)
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
    fn test_ffn_networks_1() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out_std = ffn_std.forward(&x).unwrap();
        assert_eq!(out_std.shape(), &[2, 3, 16]);

        let cfg_gated = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Swiglu,
            bias: false,
            dropout: 0.0,
        };
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 1 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }
}
