//! # Regression Evaluation Metrics
//!
//! MSE, RMSE, MAE, MAPE, SMAPE, R-squared (R²), and explained variance.
#![allow(missing_docs)]

pub mod robust;
pub use robust::{huber_metric, median_absolute_error, RobustMetricConfig};

use crate::core::{MetricError, MetricResult};
use crate::utils::stable_divide;
use brain_core::Tensor;

/// Mean Squared Error (MSE).
pub fn mse_score(preds: &Tensor, targets: &Tensor) -> MetricResult<f64> {
    if preds.shape() != targets.shape() {
        return Err(MetricError::LengthMismatch {
            expected: targets.shape()[0],
            got: preds.shape()[0],
        });
    }
    let p = preds.to_vec();
    let t = targets.to_vec();
    let n = p.len();
    if n == 0 {
        return Ok(0.0);
    }

    let sum_sq: f64 = p
        .iter()
        .zip(t.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    Ok(sum_sq / n as f64)
}

/// Root Mean Squared Error (RMSE).
pub fn rmse_score(preds: &Tensor, targets: &Tensor) -> MetricResult<f64> {
    mse_score(preds, targets).map(|v| v.sqrt())
}

/// Mean Absolute Error (MAE).
pub fn mae_score(preds: &Tensor, targets: &Tensor) -> MetricResult<f64> {
    if preds.shape() != targets.shape() {
        return Err(MetricError::LengthMismatch {
            expected: targets.shape()[0],
            got: preds.shape()[0],
        });
    }
    let p = preds.to_vec();
    let t = targets.to_vec();
    let n = p.len();
    if n == 0 {
        return Ok(0.0);
    }

    let sum_abs: f64 = p.iter().zip(t.iter()).map(|(&a, &b)| (a - b).abs()).sum();
    Ok(sum_abs / n as f64)
}

/// Coefficient of determination R² score: 1 - sum(y - y_hat)^2 / sum(y - y_bar)^2.
pub fn r2_score(preds: &Tensor, targets: &Tensor) -> MetricResult<f64> {
    if preds.shape() != targets.shape() {
        return Err(MetricError::LengthMismatch {
            expected: targets.shape()[0],
            got: preds.shape()[0],
        });
    }
    let p = preds.to_vec();
    let t = targets.to_vec();
    let n = p.len();
    if n == 0 {
        return Ok(1.0);
    }

    let mean_t: f64 = t.iter().sum::<f64>() / n as f64;
    let ss_res: f64 = p.iter().zip(t.iter()).map(|(&a, &b)| (b - a).powi(2)).sum();
    let ss_tot: f64 = t.iter().map(|&b| (b - mean_t).powi(2)).sum();

    Ok(1.0 - stable_divide(ss_res, ss_tot, 0.0))
}

/// Mean Absolute Percentage Error (MAPE).
pub fn mape_score(preds: &Tensor, targets: &Tensor) -> MetricResult<f64> {
    let p = preds.to_vec();
    let t = targets.to_vec();
    let n = p.len().min(t.len());
    if n == 0 {
        return Ok(0.0);
    }

    let sum_pe: f64 = p
        .iter()
        .zip(t.iter())
        .map(|(&a, &b)| {
            if b.abs() > 1e-12 {
                (a - b).abs() / b.abs()
            } else {
                0.0
            }
        })
        .sum();

    Ok(sum_pe / n as f64)
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
