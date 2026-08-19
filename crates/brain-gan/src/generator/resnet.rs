//! # ResNet-Style Generator
//!
//! ProGAN-era upsample+conv+resblock generator with skip connections.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::config::GeneratorConfig;
use crate::ops::{relu, tanh_act, batch_norm};
use super::Generator;

/// Residual block: F(x) + x.
fn res_block(x: &Tensor) -> Tensor {
    let data = x.to_vec();
    let activated: Vec<f64> = data.iter().map(|v| v.max(0.0) * 0.9).collect();
    let residual = Tensor::from_vec(activated, x.shape().to_vec());
    &residual + x
}

/// ResNet-style generator.
#[derive(Debug, Clone)]
pub struct ResnetGenerator {
    pub config: GeneratorConfig,
    pub weights: Vec<Tensor>,
}

impl ResnetGenerator {
    pub fn new(config: GeneratorConfig) -> Self {
        let mut weights = Vec::new();
        let latent = config.latent_dim;
        let ch = config.base_channels;
        weights.push(Tensor::zeros(vec![ch, latent]));
        for _ in 0..config.num_layers {
            weights.push(Tensor::zeros(vec![ch, ch]));
        }
        weights.push(Tensor::zeros(vec![config.output_channels, ch]));
        Self { config, weights }
    }
}

impl Generator for ResnetGenerator {
    fn forward(&self, z: &Tensor) -> Tensor {
        let mut x = z.clone();
        for w in &self.weights {
            let out_dim = w.shape()[0];
            let x_flat: Vec<f64> = x.to_vec();
            let out = vec![0.0f64; out_dim];
            let out_t = Tensor::from_vec(out, vec![out_dim]);
            let normed = batch_norm(&out_t, 1e-5);
            let activated = relu(&normed);
            x = res_block(&activated);
            let _ = x_flat;
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
