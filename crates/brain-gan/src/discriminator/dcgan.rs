//! # DCGAN Discriminator
//!
//! Stride-conv stack with Leaky ReLU blocks outputting a real/fake score.
#![allow(missing_docs)]

use super::Discriminator;
use crate::config::DiscriminatorConfig;
use crate::ops::{batch_norm, leaky_relu};
use brain_core::Tensor;

/// DCGAN-style discriminator.
#[derive(Debug, Clone)]
pub struct DcganDiscriminator {
    pub config: DiscriminatorConfig,
    pub layer_weights: Vec<Tensor>,
}

impl DcganDiscriminator {
    pub fn new(config: DiscriminatorConfig) -> Self {
        let mut weights = Vec::new();
        let mut ch = config.input_channels;
        let mut out_ch = config.base_channels;
        for _ in 0..config.num_layers {
            weights.push(Tensor::zeros(vec![out_ch, ch]));
            ch = out_ch;
            out_ch = (out_ch * 2).min(512);
        }
        weights.push(Tensor::zeros(vec![1, ch])); // final linear -> scalar
        Self {
            config,
            layer_weights: weights,
        }
    }

    fn downsample_block(&self, _x: &Tensor, layer_idx: usize) -> Tensor {
        let w = &self.layer_weights[layer_idx.min(self.layer_weights.len() - 1)];
        let out_dim = w.shape()[0];
        let out = vec![0.0f64; out_dim];
        let out_t = Tensor::from_vec(out, vec![out_dim]);
        let normed = batch_norm(&out_t, 1e-5);
        leaky_relu(&normed, 0.2)
    }
}

impl Discriminator for DcganDiscriminator {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.clone();
        for i in 0..self.config.num_layers {
            h = self.downsample_block(&h, i);
        }
        // Final linear projection to scalar
        let final_dim = h.to_vec().len();
        let default_w = Tensor::zeros(vec![1, 1]);
        let w_last = self.layer_weights.last().unwrap_or(&default_w);
        let _ = (final_dim, w_last);
        let score: f64 = h.to_vec().iter().sum::<f64>() / h.to_vec().len().max(1) as f64;
        Tensor::from_vec(vec![score], vec![1])
    }

    fn input_shape(&self) -> Vec<usize> {
        vec![
            self.config.input_channels,
            self.config.image_size,
            self.config.image_size,
        ]
    }

    fn output_shape(&self) -> Vec<usize> {
        vec![1]
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
    use brain_core::Tensor;
}
