//! # GAN Loss Functions
//!
//! Trait [`GanLoss`], `LossVariant` enum, and per-variant dispatch.
#![allow(missing_docs)]

pub mod classic;
pub mod perceptual;

pub use classic::{
    bce_loss_d, bce_loss_g, hinge_loss_d, hinge_loss_g, lsgan_loss_d, lsgan_loss_g, wgan_loss_d,
    wgan_loss_g, ClassicLoss,
};
pub use perceptual::{feature_matching_loss, gram_matrix, PerceptualConfig};

/// Trait for GAN loss functions.
pub trait GanLoss {
    fn discriminator_loss(&self, d_real: f64, d_fake: f64) -> f64;
    fn generator_loss(&self, d_fake: f64) -> f64;
}

/// Configuration for the GAN loss computation.
#[derive(Debug, Clone, Default)]
pub struct GanLossConfig {
    pub label_smoothing: f64,
    pub relativistic: bool,
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
