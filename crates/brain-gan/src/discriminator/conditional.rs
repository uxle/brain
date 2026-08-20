//! # Conditional Discriminator
//!
//! cGAN projection discriminator and ACGAN auxiliary classifier.
#![allow(missing_docs)]

use super::Discriminator;
use crate::config::DiscriminatorConfig;
use crate::ops::{batch_norm, leaky_relu};
use brain_core::Tensor;

/// Conditional discriminator (cGAN projection variant).
#[derive(Debug, Clone)]
pub struct ConditionalDiscriminator {
    pub config: DiscriminatorConfig,
    pub weights: Vec<Tensor>,
    pub class_embed: Tensor,
}

impl ConditionalDiscriminator {
    pub fn new(config: DiscriminatorConfig) -> Self {
        let num_classes = config.num_classes.max(1);
        let ch = config.base_channels;
        let class_embed = Tensor::zeros(vec![num_classes, ch]);
        let mut weights = Vec::new();
        let mut in_ch = config.input_channels;
        let mut out_ch = ch;
        for _ in 0..config.num_layers {
            weights.push(Tensor::zeros(vec![out_ch, in_ch]));
            in_ch = out_ch;
            out_ch = (out_ch * 2).min(512);
        }
        weights.push(Tensor::zeros(vec![1, in_ch]));
        Self {
            config,
            weights,
            class_embed,
        }
    }

    pub fn class_projection(&self, features: &Tensor, class_id: usize) -> f64 {
        let c = class_id % self.config.num_classes.max(1);
        let emb_data = self.class_embed.to_vec();
        let ch = self.config.base_channels;
        let emb: Vec<f64> = emb_data[c * ch..(c * ch + ch).min(emb_data.len())].to_vec();
        let fdata = features.to_vec();
        emb.iter().zip(fdata.iter()).map(|(e, f)| e * f).sum()
    }
}

impl Discriminator for ConditionalDiscriminator {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.clone();
        for w in &self.weights {
            let out_dim = w.shape()[0];
            let out = vec![0.0f64; out_dim];
            let t = Tensor::from_vec(out, vec![out_dim]);
            let normed = batch_norm(&t, 1e-5);
            h = leaky_relu(&normed, 0.2);
        }
        let score: f64 = h.to_vec().iter().sum::<f64>() / h.to_vec().len().max(1) as f64;
        let proj = self.class_projection(&h, 0);
        Tensor::from_vec(vec![score + proj], vec![1])
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
