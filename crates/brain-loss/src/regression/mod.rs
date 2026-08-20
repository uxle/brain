//! # Regression Losses
//!
//! Regression loss trait and implementations: MSE, MAE, Huber, Smooth L1, Quantile, and Directional.
#![allow(missing_docs)]

pub mod dirichlet;
pub mod mse;
pub mod robust;

pub use dirichlet::{AngularDistanceLoss, CosineEmbeddingLoss};
pub use mse::{HuberLoss, MAELoss, MSELoss, SmoothL1Loss};
pub use robust::{CauchyLoss, QuantileLoss, RobustConfig};

use crate::core::{LossResult, Reduction};
use brain_core::Tensor;

/// General configuration for regression losses.
#[derive(Debug, Clone)]
pub struct RegLossConfig {
    pub reduction: Reduction,
    pub delta: f64,
}

impl Default for RegLossConfig {
    fn default() -> Self {
        Self {
            reduction: Reduction::Mean,
            delta: 1.0,
        }
    }
}

/// Trait for regression loss algorithms.
pub trait RegressionLoss: Send + Sync {
    /// Computes regression loss given prediction and continuous target tensors.
    fn compute(&self, pred: &Tensor, target: &Tensor) -> LossResult<Tensor>;
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
