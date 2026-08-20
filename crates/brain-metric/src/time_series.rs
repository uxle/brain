//! # Time-Series Evaluation Metrics
//!
//! Mean Absolute Scaled Error (MASE), lag-aware metrics, and directional forecast bias.
#![allow(missing_docs)]

use crate::utils::stable_divide;
use brain_core::Tensor;

/// Configuration for time-series forecasting metrics.
#[derive(Debug, Clone, Default)]
pub struct TsConfig {
    pub seasonality_lag: usize,
}

/// Mean Absolute Scaled Error (MASE): MAE(forecast) / MAE(naive seasonal baseline).
pub fn mase_score(forecast: &Tensor, actual: &Tensor, seasonality_lag: usize) -> f64 {
    let f = forecast.to_vec();
    let y = actual.to_vec();
    let n = f.len().min(y.len());
    if n <= seasonality_lag {
        return 0.0;
    }

    let mae_forecast: f64 = f
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f64>()
        / n as f64;

    let mut sum_naive_diff = 0.0f64;
    let count_naive = n - seasonality_lag;
    for i in seasonality_lag..n {
        sum_naive_diff += (y[i] - y[i - seasonality_lag]).abs();
    }
    let mae_naive = sum_naive_diff / count_naive as f64;

    stable_divide(mae_forecast, mae_naive, 1.0)
}

/// Forecast Bias: mean(forecast - actual). Positive = overprediction, Negative = underprediction.
pub fn forecast_bias(forecast: &Tensor, actual: &Tensor) -> f64 {
    let f = forecast.to_vec();
    let y = actual.to_vec();
    let n = f.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    let sum_diff: f64 = f.iter().zip(y.iter()).map(|(&a, &b)| a - b).sum();
    sum_diff / n as f64
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
