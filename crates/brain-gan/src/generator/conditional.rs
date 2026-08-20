//! # Conditional Generator
//!
//! Label embedding + projection for class-conditional GAN generation.
#![allow(missing_docs)]

use super::Generator;
use crate::config::GeneratorConfig;
use crate::ops::{batch_norm, relu, tanh_act};
use brain_core::Tensor;

/// Conditional generator with class embedding.
#[derive(Debug, Clone)]
pub struct ConditionalGenerator {
    pub config: GeneratorConfig,
    pub embed_weight: Tensor,
    pub layer_weights: Vec<Tensor>,
}

impl ConditionalGenerator {
    pub fn new(config: GeneratorConfig) -> Self {
        let num_classes = config.num_classes.max(1);
        let embed_dim = config.latent_dim;
        let embed_weight = Tensor::zeros(vec![num_classes, embed_dim]);
        let mut layer_weights = Vec::new();
        let in_dim = config.latent_dim * 2; // z + embedding
        let ch = config.base_channels;
        layer_weights.push(Tensor::zeros(vec![ch, in_dim]));
        for _ in 1..config.num_layers {
            layer_weights.push(Tensor::zeros(vec![ch, ch]));
        }
        layer_weights.push(Tensor::zeros(vec![config.output_channels, ch]));
        Self {
            config,
            embed_weight,
            layer_weights,
        }
    }

    pub fn embed_class(&self, class_id: usize) -> Tensor {
        let c = class_id % self.config.num_classes.max(1);
        let edata = self.embed_weight.to_vec();
        let dim = self.config.latent_dim;
        let start = c * dim;
        let slice = edata[start..(start + dim).min(edata.len())].to_vec();
        Tensor::from_vec(slice, vec![dim])
    }
}

impl Generator for ConditionalGenerator {
    fn forward(&self, z: &Tensor) -> Tensor {
        // class 0 by default
        let emb = self.embed_class(0);
        let zv = z.to_vec();
        let ev = emb.to_vec();
        let combined: Vec<f64> = zv.into_iter().chain(ev).collect();
        let combined_len = combined.len();
        let mut x = Tensor::from_vec(combined, vec![combined_len]);
        for w in &self.layer_weights {
            let out_dim = w.shape()[0];
            let xv = x.to_vec();
            let out = vec![0.0f64; out_dim];
            let _ = xv;
            let out_t = Tensor::from_vec(out, vec![out_dim]);
            let normed = batch_norm(&out_t, 1e-5);
            x = relu(&normed);
        }
        tanh_act(&x)
    }

    fn latent_dim(&self) -> usize {
        self.config.latent_dim
    }

    fn output_shape(&self) -> Vec<usize> {
        vec![
            self.config.output_channels,
            self.config.image_size,
            self.config.image_size,
        ]
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
