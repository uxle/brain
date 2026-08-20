//! # Wasserstein GAN (WGAN) Loss
//!
//! Earth Mover's Distance objective: L_D = -E[D(x)] + E[D(G(z))], L_G = -E[D(G(z))].
#![allow(missing_docs)]

use super::AdversarialLoss;
use crate::core::LossResult;
use brain_core::Tensor;

/// Configuration for WGAN.
#[derive(Debug, Clone, Default)]
pub struct WassersteinConfig {
    pub gradient_penalty_weight: f64,
}

/// Wasserstein GAN loss module.
#[derive(Debug, Clone, Default)]
pub struct WassersteinLoss {
    pub config: WassersteinConfig,
}

impl AdversarialLoss for WassersteinLoss {
    fn discriminator_loss(&self, d_real: &Tensor, d_fake: &Tensor) -> LossResult<Tensor> {
        let r_mean = d_real.to_vec().iter().sum::<f64>() / d_real.to_vec().len().max(1) as f64;
        let f_mean = d_fake.to_vec().iter().sum::<f64>() / d_fake.to_vec().len().max(1) as f64;
        Ok(Tensor::from_vec(vec![-(r_mean - f_mean)], vec![1]))
    }

    fn generator_loss(&self, d_fake: &Tensor) -> LossResult<Tensor> {
        let f_mean = d_fake.to_vec().iter().sum::<f64>() / d_fake.to_vec().len().max(1) as f64;
        Ok(Tensor::from_vec(vec![-f_mean], vec![1]))
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
