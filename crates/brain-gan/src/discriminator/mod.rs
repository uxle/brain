//! # Discriminator Module
//!
//! [`Discriminator`] trait, `DiscriminatorConfig`, score interpretation.
#![allow(missing_docs)]

pub mod conditional;
pub mod dcgan;
pub mod patch;

pub use conditional::ConditionalDiscriminator;
pub use dcgan::DcganDiscriminator;
pub use patch::PatchDiscriminator;

use brain_core::Tensor;

/// Core trait for GAN discriminators.
pub trait Discriminator: Send + Sync {
    /// Forward pass: maps image to a real/fake score.
    fn forward(&self, x: &Tensor) -> Tensor;
    /// Returns expected input shape [C, H, W].
    fn input_shape(&self) -> Vec<usize>;
    /// Returns the output scalar (or patch grid) shape.
    fn output_shape(&self) -> Vec<usize>;
}

/// Interprets a discriminator output as a probability via sigmoid.
pub fn score_to_prob(score: f64) -> f64 {
    1.0 / (1.0 + (-score).exp())
}

/// Interprets a batch discriminator output as probability vector.
pub fn batch_score_to_prob(scores: &Tensor) -> Tensor {
    let data: Vec<f64> = scores.to_vec().iter().map(|&s| score_to_prob(s)).collect();
    Tensor::from_vec(data, scores.shape().to_vec())
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
    use crate::config::DiscriminatorConfig;
    use brain_core::Tensor;
}
