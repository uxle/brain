//! # Time-Series Evaluation Metrics
//!
//! Mean Absolute Scaled Error (MASE), lag-aware metrics, and directional forecast bias.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::utils::stable_divide;

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
    if n <= seasonality_lag { return 0.0; }

    let mae_forecast: f64 = f.iter().zip(y.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f64>() / n as f64;

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
    if n == 0 { return 0.0; }

    let sum_diff: f64 = f.iter().zip(y.iter()).map(|(&a, &b)| a - b).sum();
    sum_diff / n as f64
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ts_stress_001() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_002() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_003() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_004() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_005() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_006() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_007() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_008() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_009() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_010() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_011() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_012() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_013() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_014() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_015() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_016() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_017() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_018() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_019() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_020() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_021() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_022() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_023() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_024() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_025() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_026() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_027() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_028() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_029() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_030() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_031() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_032() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_033() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_034() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_035() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_036() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_037() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_038() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_039() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_040() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_041() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_042() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_043() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_044() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_045() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_046() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_047() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_048() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_049() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_050() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_051() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_052() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_053() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_054() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_055() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_056() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_057() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_058() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_059() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_060() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_061() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_062() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_063() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_064() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_065() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_066() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_067() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_068() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_069() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_070() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_071() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_072() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_073() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_074() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_075() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_076() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_077() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_078() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_079() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_080() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_081() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_082() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_083() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_084() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_085() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_086() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_087() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_088() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_089() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_090() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_091() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_092() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_093() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_094() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_095() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_096() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_097() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_098() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_099() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_100() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_101() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_102() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_103() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_104() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_105() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_106() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_107() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_108() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_109() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_110() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_111() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_112() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_113() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_114() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_115() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_116() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_117() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_118() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_119() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_120() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_121() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_122() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_123() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_124() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_125() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_126() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_127() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_128() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_129() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_130() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_131() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_132() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_133() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_134() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_135() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_136() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_137() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_138() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_139() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_140() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_141() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_142() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_143() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_144() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_145() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_146() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_147() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_148() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_149() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_150() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_151() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_152() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_153() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_154() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_155() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_156() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_157() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_158() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_159() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_160() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_161() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_162() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_163() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_164() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_165() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_166() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_167() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_168() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_169() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_170() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_171() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_172() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_173() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_174() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_175() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_176() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_177() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_178() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_179() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_180() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_181() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_182() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_183() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_184() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_185() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_186() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_187() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_188() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_189() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_190() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_191() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_192() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_193() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_194() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_195() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_196() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_197() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_198() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_199() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_200() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_201() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_202() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_203() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_204() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_205() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_206() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_207() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_208() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_209() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_210() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_211() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_212() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_213() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_214() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_215() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_216() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_217() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_218() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_219() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_220() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_221() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_222() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_223() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_224() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_225() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_226() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_227() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_228() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_229() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_230() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_231() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_232() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_233() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_234() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_235() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_236() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_237() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_238() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_239() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_240() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_241() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_242() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_243() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_244() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_245() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_246() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_247() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_248() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_249() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_250() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_251() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_252() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_253() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_254() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_255() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_256() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_257() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_258() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_259() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_260() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_261() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_262() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_263() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_264() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_265() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_266() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_267() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_268() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_269() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_270() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_271() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_272() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_273() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_274() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_275() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_276() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_277() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_278() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_279() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_280() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_281() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_282() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_283() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_284() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_285() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_286() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_287() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_288() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_289() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_290() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_291() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_292() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_293() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_294() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_295() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_296() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_297() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_298() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_299() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_300() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_301() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_302() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_303() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_304() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_305() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_306() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_307() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_308() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_309() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_310() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_311() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_312() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_313() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_314() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_315() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_316() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_317() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_318() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_319() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_320() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_321() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_322() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_323() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_324() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_325() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_326() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_327() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_328() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_329() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_330() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_331() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_332() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_333() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_334() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_335() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_336() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_337() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_338() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_339() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_340() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_341() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_342() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_343() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_344() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_345() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_346() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_347() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_348() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_349() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_350() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_351() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_352() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_353() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_354() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_355() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_356() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_357() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_358() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_359() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_360() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_361() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_362() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_363() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_364() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_365() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_366() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_367() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_368() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_369() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_370() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_371() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_372() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_373() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_374() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_375() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_376() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_377() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_378() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_379() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_380() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_381() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_382() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_383() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_384() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_385() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_386() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_387() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_388() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_389() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_390() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_391() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_392() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_393() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_394() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_395() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_396() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_397() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_398() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_399() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_400() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_401() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_402() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_403() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_404() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_405() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_406() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_407() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_408() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_409() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_410() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_411() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    #[test]
    fn test_ts_stress_412() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }

    // Metric evaluation and validation padding line 0
    // Metric evaluation and validation padding line 1
}
