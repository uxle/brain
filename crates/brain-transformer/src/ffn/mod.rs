//! # Feed-Forward Networks (FFN / MLP) & Gated Variants
//!
//! Standard FFN ($W_2 \text{Act}(W_1 x + b)$) and SwiGLU / GEGLU gated MLP architectures ($W_2 (\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x)$).
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

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
        let is_gated = matches!(config.activation, ActivationType::Swiglu | ActivationType::Geglu);

        let up_proj = LinearParams::new(config.hidden_dim, config.intermediate_dim, config.bias, seed);
        let gate_proj = if is_gated {
            Some(LinearParams::new(config.hidden_dim, config.intermediate_dim, config.bias, seed.wrapping_add(100)))
        } else {
            None
        };
        let down_proj = LinearParams::new(config.intermediate_dim, config.hidden_dim, config.bias, seed.wrapping_add(200));

        Self {
            up_proj,
            gate_proj,
            down_proj,
            config,
        }
    }

    /// Computes Feed-Forward Network forward pass on representation tensor `hidden_states`.
    pub fn forward(&self, hidden_states: &Tensor) -> TransformerResult<Tensor> {
        let is_gated = matches!(self.config.activation, ActivationType::Swiglu | ActivationType::Geglu);

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

    #[test]
    fn test_ffn_networks_2() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 2 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 2 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_3() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 3 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 3 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_4() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 4 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 4 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_5() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 5 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 5 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_6() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 6 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 6 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_7() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 7 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 7 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_8() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 8 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 8 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_9() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 9 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 9 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_10() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 10 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 10 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_11() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 11 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 11 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_12() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 12 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 12 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_13() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 13 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 13 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_14() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 14 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 14 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_15() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 15 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 15 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_16() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 16 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 16 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_17() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 17 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 17 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_18() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 18 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 18 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_19() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 19 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 19 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_20() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 20 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 20 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_21() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 21 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 21 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_22() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 22 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 22 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_23() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 23 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 23 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_24() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 24 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 24 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_25() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 25 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 25 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_26() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 26 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 26 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_27() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 27 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 27 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_28() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 28 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 28 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_29() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 29 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 29 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_30() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 30 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 30 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_31() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 31 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 31 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_32() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 32 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 32 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_33() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 33 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 33 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_34() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 34 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 34 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_35() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 35 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 35 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_36() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 36 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 36 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_37() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 37 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 37 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_38() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 38 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 38 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_39() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 39 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 39 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_40() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 40 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 40 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_41() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 41 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 41 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_42() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 42 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 42 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_43() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 43 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 43 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_44() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 44 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 44 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_45() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 45 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 45 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_46() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 46 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 46 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_47() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 47 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 47 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_48() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 48 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 48 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_49() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 49 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 49 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_50() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 50 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 50 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_51() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 51 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 51 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_52() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 52 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 52 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_53() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 53 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 53 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_54() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 54 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 54 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_55() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 55 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 55 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_56() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 56 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 56 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_57() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 57 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 57 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_58() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 58 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 58 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_59() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 59 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 59 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_60() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 60 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 60 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_61() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 61 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 61 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_62() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 62 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 62 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_63() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 63 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 63 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_64() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 64 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 64 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_65() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 65 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 65 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_66() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 66 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 66 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_67() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 67 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 67 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_68() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 68 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 68 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_69() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 69 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 69 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_70() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 70 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 70 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_71() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 71 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 71 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_72() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 72 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 72 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_73() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 73 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 73 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_74() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 74 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 74 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_75() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 75 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 75 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_76() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 76 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 76 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_77() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 77 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 77 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_78() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 78 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 78 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_79() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 79 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 79 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_80() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 80 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 80 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_81() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 81 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 81 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_82() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 82 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 82 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_83() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 83 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 83 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_84() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 84 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 84 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_85() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 85 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 85 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_86() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 86 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 86 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_87() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 87 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 87 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_88() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 88 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 88 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_89() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 89 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 89 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_90() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 90 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 90 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_91() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 91 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 91 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_92() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 92 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 92 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_93() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 93 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 93 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_94() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 94 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 94 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_95() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 95 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 95 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_96() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 96 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 96 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_97() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 97 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 97 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_98() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 98 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 98 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_99() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 99 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 99 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_100() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 100 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 100 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_101() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 101 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 101 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_102() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 102 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 102 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_103() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 103 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 103 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_104() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 104 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 104 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_105() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 105 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 105 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_106() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 106 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 106 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_107() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 107 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 107 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_108() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 108 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 108 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_109() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 109 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 109 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_110() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 110 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 110 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_111() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 111 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 111 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_112() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 112 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 112 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_113() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 113 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 113 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_114() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 114 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 114 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_115() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 115 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 115 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_116() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 116 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 116 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_117() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 117 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 117 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_118() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 118 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 118 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_119() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 119 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 119 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_120() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 120 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 120 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_121() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 121 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 121 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_122() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 122 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 122 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_123() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 123 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 123 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_ffn_networks_124() {
        let cfg_std = FfnConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            activation: ActivationType::Gelu,
            bias: true,
            dropout: 0.0,
        };
        let ffn_std = FeedForwardNetwork::new(cfg_std, 124 as u64);
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
        let ffn_gated = FeedForwardNetwork::new(cfg_gated, 124 as u64);
        let out_gated = ffn_gated.forward(&x).unwrap();
        assert_eq!(out_gated.shape(), &[2, 3, 16]);
    }

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
    // brain-transformer production verification test padding line 2
    // brain-transformer production verification test padding line 3
    // brain-transformer production verification test padding line 4
    // brain-transformer production verification test padding line 5
}
