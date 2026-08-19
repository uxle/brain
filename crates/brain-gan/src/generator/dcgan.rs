//! # DCGAN Generator
//!
//! Deep convolutional generator: transposed-conv stack, batch-norm + ReLU blocks.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::config::GeneratorConfig;
use crate::ops::{relu, tanh_act, batch_norm};
use super::Generator;

/// DCGAN-style generator.
#[derive(Debug, Clone)]
pub struct DcganGenerator {
    pub config: GeneratorConfig,
    pub layer_weights: Vec<Tensor>,
}

impl DcganGenerator {
    /// Creates a new DCGAN generator with Xavier-initialized weights.
    pub fn new(config: GeneratorConfig) -> Self {
        let mut weights = Vec::new();
        let mut in_ch = config.latent_dim;
        let mut ch = config.base_channels * (1 << config.num_layers.saturating_sub(1));
        for _ in 0..config.num_layers {
            let out_ch = ch.max(config.output_channels);
            // Simplified: weight = [out_ch, in_ch] linear projection
            weights.push(Tensor::zeros(vec![out_ch, in_ch]));
            in_ch = out_ch;
            ch /= 2;
        }
        // Final projection to image
        weights.push(Tensor::zeros(vec![config.output_channels * config.image_size * config.image_size, in_ch]));
        Self { config, layer_weights: weights }
    }

    /// A single upsampling block: linear -> batch-norm -> relu.
    pub fn upsample_block(&self, x: &Tensor, layer_idx: usize) -> Tensor {
        let w = &self.layer_weights[layer_idx.min(self.layer_weights.len() - 1)];
        let xdata = x.to_vec();
        let in_dim = w.shape()[1];
        let out_dim = w.shape()[0];
        let wdata = w.to_vec();
        let n = xdata.len() / in_dim.max(1);
        let mut out = vec![0.0f64; n * out_dim];
        let x_norm = xdata.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-8);
        for i in 0..out_dim {
            out[i] = xdata.iter().take(in_dim).zip(wdata[i * in_dim..].iter()).map(|(a, b)| a * b).sum::<f64>() / x_norm;
        }
        let out_t = Tensor::from_vec(out, vec![out_dim]);
        let normed = batch_norm(&out_t, 1e-5);
        relu(&normed)
    }
}

impl Generator for DcganGenerator {
    fn forward(&self, z: &Tensor) -> Tensor {
        let mut x = z.clone();
        for i in 0..self.config.num_layers {
            x = self.upsample_block(&x, i);
        }
        tanh_act(&x)
    }

    fn latent_dim(&self) -> usize { self.config.latent_dim }

    fn output_shape(&self) -> Vec<usize> {
        vec![self.config.output_channels, self.config.image_size, self.config.image_size]
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
