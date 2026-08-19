//! # Adversarial Loss Functions
//!
//! Minimax, Wasserstein (WGAN), Hinge, Least Squares (LSGAN), and Relativistic GAN losses.
#![allow(missing_docs)]

pub mod wasserstein;
pub mod other;

pub use wasserstein::{WassersteinLoss, WassersteinConfig};
pub use other::{HingeAdversarialLoss, LSGANLoss, RelativisticLoss, AdvLossKind};

use brain_core::Tensor;
use crate::core::LossResult;

/// Configuration for adversarial loss objectives.
#[derive(Debug, Clone, Default)]
pub struct AdvLossConfig {
    pub label_smoothing: f64,
    pub gp_lambda: f64,
}

/// Trait for Generative Adversarial Network discriminator and generator losses.
pub trait AdversarialLoss: Send + Sync {
    /// Computes discriminator loss from real and fake prediction scores.
    fn discriminator_loss(&self, d_real: &Tensor, d_fake: &Tensor) -> LossResult<Tensor>;
    /// Computes generator loss from fake prediction scores.
    fn generator_loss(&self, d_fake: &Tensor) -> LossResult<Tensor>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
