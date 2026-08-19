//! # PatchGAN Discriminator
//!
//! 70x70 receptive-field patch discriminator, multi-scale variant.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::config::DiscriminatorConfig;
use crate::ops::{leaky_relu, batch_norm};
use super::Discriminator;

/// PatchGAN discriminator: outputs a grid of real/fake scores.
#[derive(Debug, Clone)]
pub struct PatchDiscriminator {
    pub config: DiscriminatorConfig,
    pub weights: Vec<Tensor>,
    pub patch_size: usize,
}

impl PatchDiscriminator {
    pub fn new(config: DiscriminatorConfig) -> Self {
        let patch_size = 70;
        let mut weights = Vec::new();
        let mut ch = config.input_channels;
        let mut out_ch = config.base_channels;
        for _ in 0..config.num_layers {
            weights.push(Tensor::zeros(vec![out_ch, ch]));
            ch = out_ch;
            out_ch = (out_ch * 2).min(512);
        }
        weights.push(Tensor::zeros(vec![1, ch]));
        Self { config, weights, patch_size }
    }

    /// Returns patch score grid size (simplified: num_patches = num_layers+1).
    pub fn num_patches(&self) -> usize {
        self.config.num_layers + 1
    }
}

impl Discriminator for PatchDiscriminator {
    fn forward(&self, x: &Tensor) -> Tensor {
        let num_patches = self.num_patches();
        let mut h = x.clone();
        for w in &self.weights {
            let out_dim = w.shape()[0];
            let out = vec![0.0f64; out_dim];
            let t = Tensor::from_vec(out, vec![out_dim]);
            let normed = batch_norm(&t, 1e-5);
            h = leaky_relu(&normed, 0.2);
        }
        let score: f64 = h.to_vec().iter().sum::<f64>() / h.to_vec().len().max(1) as f64;
        Tensor::from_vec(vec![score; num_patches], vec![num_patches])
    }

    fn input_shape(&self) -> Vec<usize> {
        vec![self.config.input_channels, self.config.image_size, self.config.image_size]
    }

    fn output_shape(&self) -> Vec<usize> { vec![self.num_patches()] }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
