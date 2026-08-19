//! # Classic Regression Losses
//!
//! Mean Squared Error (MSE), Mean Absolute Error (MAE), Huber Loss, Log-Cosh, and Smooth L1.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{LossResult, Reduction};
use crate::utils::{reduction_apply, check_shapes};
use super::RegressionLoss;

/// Mean Squared Error (L2 loss).
#[derive(Debug, Clone, Default)]
pub struct MSELoss {
    pub reduction: Reduction,
}

impl RegressionLoss for MSELoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> LossResult<Tensor> {
        check_shapes(pred, target)?;
        let diff = pred - target;
        let sq = &diff * &diff;
        Ok(reduction_apply(&sq.to_vec(), self.reduction))
    }
}

/// Mean Absolute Error (L1 loss).
#[derive(Debug, Clone, Default)]
pub struct MAELoss {
    pub reduction: Reduction,
}

impl RegressionLoss for MAELoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> LossResult<Tensor> {
        check_shapes(pred, target)?;
        let diff = pred - target;
        let abs_vals: Vec<f64> = diff.to_vec().iter().map(|&v| v.abs()).collect();
        Ok(reduction_apply(&abs_vals, self.reduction))
    }
}

/// Huber Loss (smooth transition between L2 and L1 at delta threshold).
#[derive(Debug, Clone)]
pub struct HuberLoss {
    pub delta: f64,
    pub reduction: Reduction,
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self { delta: 1.0, reduction: Reduction::Mean }
    }
}

impl RegressionLoss for HuberLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> LossResult<Tensor> {
        check_shapes(pred, target)?;
        let diff = pred - target;
        let d = self.delta;
        let losses: Vec<f64> = diff.to_vec().iter().map(|&x| {
            let abs_x = x.abs();
            if abs_x <= d {
                0.5 * abs_x * abs_x
            } else {
                d * (abs_x - 0.5 * d)
            }
        }).collect();

        Ok(reduction_apply(&losses, self.reduction))
    }
}

/// Smooth L1 Loss (alias for Huber with delta = 1.0 or custom beta).
pub type SmoothL1Loss = HuberLoss;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
