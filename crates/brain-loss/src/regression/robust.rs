//! # Robust & Quantile Losses
//!
//! Pinball / Quantile loss, Log-Huber, and Cauchy loss for outlier-resistant regression.
#![allow(missing_docs)]

use super::RegressionLoss;
use crate::core::{LossResult, Reduction};
use crate::utils::{check_shapes, reduction_apply};
use brain_core::Tensor;

/// Configuration for robust regression estimators.
#[derive(Debug, Clone)]
pub struct RobustConfig {
    pub tau: f64,
    pub c: f64,
    pub reduction: Reduction,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self {
            tau: 0.5,
            c: 1.0,
            reduction: Reduction::Mean,
        }
    }
}

/// Quantile / Pinball loss for conditional quantile estimation: L_tau(u) = u * (tau - I(u < 0)).
#[derive(Debug, Clone)]
pub struct QuantileLoss {
    pub tau: f64,
    pub reduction: Reduction,
}

impl Default for QuantileLoss {
    fn default() -> Self {
        Self {
            tau: 0.5,
            reduction: Reduction::Mean,
        }
    }
}

impl RegressionLoss for QuantileLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> LossResult<Tensor> {
        check_shapes(pred, target)?;
        let diff = target - pred; // u = y - y_hat
        let tau = self.tau;
        let losses: Vec<f64> = diff
            .to_vec()
            .iter()
            .map(|&u| if u >= 0.0 { tau * u } else { (tau - 1.0) * u })
            .collect();

        Ok(reduction_apply(&losses, self.reduction))
    }
}

/// Cauchy Loss: L(r) = (c^2 / 2) * ln(1 + (r / c)^2).
#[derive(Debug, Clone)]
pub struct CauchyLoss {
    pub c: f64,
    pub reduction: Reduction,
}

impl Default for CauchyLoss {
    fn default() -> Self {
        Self {
            c: 1.0,
            reduction: Reduction::Mean,
        }
    }
}

impl RegressionLoss for CauchyLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> LossResult<Tensor> {
        check_shapes(pred, target)?;
        let diff = pred - target;
        let c = self.c;
        let c2 = c * c;
        let losses: Vec<f64> = diff
            .to_vec()
            .iter()
            .map(|&r| 0.5 * c2 * (1.0 + (r * r) / c2).ln())
            .collect();

        Ok(reduction_apply(&losses, self.reduction))
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
