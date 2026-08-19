//! # Generator Module
//!
//! [`Generator`] trait, latent sampling, output range handling.
#![allow(missing_docs)]

pub mod dcgan;
pub mod resnet;
pub mod conditional;

pub use dcgan::DcganGenerator;
pub use resnet::ResnetGenerator;
pub use conditional::ConditionalGenerator;

use brain_core::Tensor;
use crate::config::{GeneratorConfig, LatentType, OutputActivation};
use crate::utils::sample_gaussian;
use crate::ops::{tanh_act, sigmoid_act};

/// Core trait for all GAN generators.
pub trait Generator: Send + Sync {
    /// Forward pass: maps latent vector `z` to a generated image tensor.
    fn forward(&self, z: &Tensor) -> Tensor;
    /// Returns latent dimension.
    fn latent_dim(&self) -> usize;
    /// Returns output shape [C, H, W].
    fn output_shape(&self) -> Vec<usize>;
}

/// Samples a latent vector according to the configured distribution.
pub fn sample_latent(config: &GeneratorConfig, seed: u64) -> Tensor {
    match config.latent_type {
        LatentType::Gaussian => {
            let data = sample_gaussian(config.latent_dim, seed);
            Tensor::from_vec(data, vec![config.latent_dim])
        }
        LatentType::Uniform => {
            let mut rng = seed;
            let data: Vec<f64> = (0..config.latent_dim).map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (rng >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            }).collect();
            Tensor::from_vec(data, vec![config.latent_dim])
        }
        LatentType::Spherical => {
            let data = sample_gaussian(config.latent_dim, seed);
            let norm = data.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-8);
            let normalized: Vec<f64> = data.iter().map(|v| v / norm).collect();
            Tensor::from_vec(normalized, vec![config.latent_dim])
        }
    }
}

/// Applies the output activation to a generator's output.
pub fn apply_output_activation(t: &Tensor, activation: OutputActivation) -> Tensor {
    match activation {
        OutputActivation::Tanh => tanh_act(t),
        OutputActivation::Sigmoid => sigmoid_act(t),
        OutputActivation::Linear => t.clone(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
