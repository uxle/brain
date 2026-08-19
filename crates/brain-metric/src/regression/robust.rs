//! # Robust Regression Metrics
//!
//! Median Absolute Error, Huber error, and quantile loss metrics.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for robust metric calculation.
#[derive(Debug, Clone, Default)]
pub struct RobustMetricConfig {
    pub huber_delta: f64,
}

/// Median Absolute Error: median(|y_true - y_pred|).
pub fn median_absolute_error(preds: &Tensor, targets: &Tensor) -> f64 {
    let p = preds.to_vec();
    let t = targets.to_vec();
    let n = p.len().min(t.len());
    if n == 0 { return 0.0; }

    let mut abs_diffs: Vec<f64> = p.iter().zip(t.iter()).map(|(&a, &b)| (a - b).abs()).collect();
    abs_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if n % 2 == 1 {
        abs_diffs[n / 2]
    } else {
        (abs_diffs[n / 2 - 1] + abs_diffs[n / 2]) * 0.5
    }
}

/// Huber metric error: evaluated over residuals with threshold delta.
pub fn huber_metric(preds: &Tensor, targets: &Tensor, delta: f64) -> f64 {
    let p = preds.to_vec();
    let t = targets.to_vec();
    let n = p.len().min(t.len());
    if n == 0 { return 0.0; }

    let sum: f64 = p.iter().zip(t.iter()).map(|(&a, &b)| {
        let abs_r = (a - b).abs();
        if abs_r <= delta {
            0.5 * abs_r * abs_r
        } else {
            delta * (abs_r - 0.5 * delta)
        }
    }).sum();

    sum / n as f64
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
