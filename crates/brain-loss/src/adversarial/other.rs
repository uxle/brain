//! # Hinge, LSGAN, and Relativistic GAN Losses
//!
//! Geometric and least-squares adversarial formulations.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::LossResult;
use super::AdversarialLoss;

/// Adversarial loss kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdvLossKind {
    #[default]
    Hinge,
    LeastSquares,
    Relativistic,
}

/// Hinge Adversarial Loss (Geometric GAN).
#[derive(Debug, Clone, Default)]
pub struct HingeAdversarialLoss;

impl AdversarialLoss for HingeAdversarialLoss {
    fn discriminator_loss(&self, d_real: &Tensor, d_fake: &Tensor) -> LossResult<Tensor> {
        let r_loss: f64 = d_real.to_vec().iter().map(|&x| (1.0 - x).max(0.0)).sum::<f64>() / d_real.to_vec().len() as f64;
        let f_loss: f64 = d_fake.to_vec().iter().map(|&x| (1.0 + x).max(0.0)).sum::<f64>() / d_fake.to_vec().len() as f64;
        Ok(Tensor::from_vec(vec![r_loss + f_loss], vec![1]))
    }

    fn generator_loss(&self, d_fake: &Tensor) -> LossResult<Tensor> {
        let f_mean = d_fake.to_vec().iter().sum::<f64>() / d_fake.to_vec().len().max(1) as f64;
        Ok(Tensor::from_vec(vec![-f_mean], vec![1]))
    }
}

/// Least Squares GAN (LSGAN).
#[derive(Debug, Clone, Default)]
pub struct LSGANLoss;

impl AdversarialLoss for LSGANLoss {
    fn discriminator_loss(&self, d_real: &Tensor, d_fake: &Tensor) -> LossResult<Tensor> {
        let r_loss: f64 = d_real.to_vec().iter().map(|&x| 0.5 * (x - 1.0).powi(2)).sum::<f64>() / d_real.to_vec().len() as f64;
        let f_loss: f64 = d_fake.to_vec().iter().map(|&x| 0.5 * x.powi(2)).sum::<f64>() / d_fake.to_vec().len() as f64;
        Ok(Tensor::from_vec(vec![r_loss + f_loss], vec![1]))
    }

    fn generator_loss(&self, d_fake: &Tensor) -> LossResult<Tensor> {
        let g_loss: f64 = d_fake.to_vec().iter().map(|&x| 0.5 * (x - 1.0).powi(2)).sum::<f64>() / d_fake.to_vec().len() as f64;
        Ok(Tensor::from_vec(vec![g_loss], vec![1]))
    }
}

/// Relativistic Average GAN (RaGAN).
#[derive(Debug, Clone, Default)]
pub struct RelativisticLoss;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
