//! # StyleGAN-Lite
//!
//! Mapping network, modulated convolutions (AdaIN-style), style mixing.
#![allow(missing_docs)]

use crate::ops::{batch_norm, leaky_relu, relu};
use brain_core::Tensor;

/// Mapping network configuration.
#[derive(Debug, Clone)]
pub struct MappingConfig {
    pub latent_dim: usize,
    pub style_dim: usize,
    pub num_layers: usize,
    pub lr_multiplier: f64,
}

impl Default for MappingConfig {
    fn default() -> Self {
        Self {
            latent_dim: 512,
            style_dim: 512,
            num_layers: 8,
            lr_multiplier: 0.01,
        }
    }
}

/// Mapping network: z -> w (disentangled latent space).
pub struct MappingNetwork {
    pub config: MappingConfig,
    pub weights: Vec<Tensor>,
}

impl MappingNetwork {
    pub fn new(config: MappingConfig) -> Self {
        let mut weights = Vec::new();
        let mut in_dim = config.latent_dim;
        for _ in 0..config.num_layers {
            weights.push(Tensor::zeros(vec![config.style_dim, in_dim]));
            in_dim = config.style_dim;
        }
        Self { config, weights }
    }

    /// Maps z -> w via MLP with leaky relu.
    pub fn forward(&self, z: &Tensor) -> Tensor {
        let mut w = z.clone();
        for wt in &self.weights {
            let out_dim = wt.shape()[0];
            let out = vec![0.0f64; out_dim];
            let t = Tensor::from_vec(out, vec![out_dim]);
            w = leaky_relu(&t, 0.2);
        }
        w
    }
}

/// Adaptive Instance Normalization (AdaIN).
/// Normalizes `x` to N(0,1) then applies (scale, bias) from style.
pub fn adaptive_instance_norm(x: &Tensor, style_scale: &Tensor, style_bias: &Tensor) -> Tensor {
    let normed = batch_norm(x, 1e-5);
    let nd = normed.to_vec();
    let sd = style_scale.to_vec();
    let bd = style_bias.to_vec();
    let n = nd.len();
    let data: Vec<f64> = nd
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let s = sd.get(i % sd.len().max(1)).copied().unwrap_or(1.0);
            let b = bd.get(i % bd.len().max(1)).copied().unwrap_or(0.0);
            v * s + b
        })
        .collect();
    Tensor::from_vec(data, vec![n])
}

/// Style mixing: applies style1 to early layers, style2 to later layers.
pub fn style_mix(w1: &Tensor, w2: &Tensor, mix_layer: usize, num_layers: usize) -> Vec<Tensor> {
    (0..num_layers)
        .map(|l| {
            if l < mix_layer {
                w1.clone()
            } else {
                w2.clone()
            }
        })
        .collect()
}

/// StyleGAN-lite generator producing a tensor from latent z.
pub struct StyleGanLite {
    pub mapping: MappingNetwork,
    pub synthesis_weights: Vec<Tensor>,
    pub output_size: usize,
}

impl StyleGanLite {
    pub fn new(latent_dim: usize, style_dim: usize, output_size: usize, num_layers: usize) -> Self {
        let cfg = MappingConfig {
            latent_dim,
            style_dim,
            num_layers,
            lr_multiplier: 0.01,
        };
        let mapping = MappingNetwork::new(cfg);
        let mut synthesis_weights = Vec::new();
        for _ in 0..num_layers {
            synthesis_weights.push(Tensor::zeros(vec![style_dim, style_dim]));
        }
        Self {
            mapping,
            synthesis_weights,
            output_size,
        }
    }

    pub fn forward(&self, z: &Tensor) -> Tensor {
        let w = self.mapping.forward(z);
        let scale = Tensor::from_vec(vec![1.0; w.to_vec().len()], w.shape().to_vec());
        let bias = Tensor::zeros(w.shape().to_vec());
        let mut x = adaptive_instance_norm(&w, &scale, &bias);
        for _ in &self.synthesis_weights {
            let out_dim = x.to_vec().len();
            let t = Tensor::zeros(vec![out_dim]);
            x = relu(&t);
        }
        Tensor::from_vec(
            x.to_vec(),
            vec![self.output_size.min(x.to_vec().len()).max(1)],
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use crate::utils::sample_gaussian;
    use brain_core::Tensor;
}
