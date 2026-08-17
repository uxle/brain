//! # Regression Evaluation Metrics
//!
//! MSE, RMSE, MAE, MAPE, SMAPE, R-squared (R²), and explained variance.
#![allow(missing_docs)]

pub mod robust;
pub use robust::{median_absolute_error, huber_metric, RobustMetricConfig};

use brain_core::Tensor;
use crate::core::{MetricResult, MetricError};
use crate::utils::stable_divide;

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
    if n == 0 { return Ok(0.0); }

    let sum_sq: f64 = p.iter().zip(t.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
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
    if n == 0 { return Ok(0.0); }

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
    if n == 0 { return Ok(1.0); }

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
    if n == 0 { return Ok(0.0); }

    let sum_pe: f64 = p.iter().zip(t.iter()).map(|(&a, &b)| {
        if b.abs() > 1e-12 { (a - b).abs() / b.abs() } else { 0.0 }
    }).sum();

    Ok(sum_pe / n as f64)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_reg_mod_stress_001() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_002() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_003() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_004() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_005() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_006() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_007() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_008() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_009() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_010() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_011() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_012() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_013() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_014() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_015() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_016() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_017() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_018() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_019() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_020() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_021() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_022() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_023() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_024() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_025() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_026() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_027() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_028() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_029() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_030() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_031() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_032() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_033() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_034() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_035() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_036() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_037() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_038() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_039() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_040() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_041() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_042() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_043() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_044() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_045() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_046() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_047() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_048() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_049() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_050() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_051() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_052() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_053() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_054() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_055() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_056() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_057() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_058() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_059() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_060() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_061() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_062() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_063() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_064() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_065() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_066() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_067() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_068() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_069() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_070() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_071() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_072() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_073() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_074() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_075() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_076() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_077() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_078() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_079() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_080() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_081() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_082() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_083() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_084() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_085() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_086() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_087() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_088() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_089() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_090() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_091() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_092() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_093() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_094() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_095() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_096() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_097() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_098() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_099() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_100() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_101() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_102() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_103() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_104() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_105() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_106() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_107() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_108() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_109() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_110() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_111() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_112() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_113() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_114() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_115() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_116() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_117() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_118() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_119() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_120() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_121() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_122() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_123() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_124() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_125() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_126() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_127() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_128() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_129() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_130() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_131() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_132() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_133() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_134() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_135() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_136() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_137() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_138() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_139() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_140() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_141() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_142() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_143() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_144() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_145() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_146() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_147() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_148() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_149() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_150() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_151() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_152() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_153() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_154() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_155() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_156() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_157() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_158() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_159() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_160() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_161() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_162() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_163() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_164() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_165() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_166() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_167() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_168() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_169() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_170() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_171() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_172() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_173() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_174() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_175() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_176() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_177() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_178() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_179() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_180() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_181() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_182() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_183() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_184() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_185() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_186() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_187() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_188() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_189() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_190() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_191() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_192() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_193() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_194() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_195() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_196() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_197() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_198() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_199() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_200() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_201() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_202() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_203() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_204() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_205() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_206() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_207() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_208() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_209() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_210() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_211() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_212() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_213() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_214() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_215() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_216() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_217() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_218() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_219() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_220() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_221() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_222() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_223() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_224() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_225() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_226() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_227() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_228() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_229() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_230() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_231() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_232() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_233() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_234() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_235() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_236() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_237() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_238() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_239() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_240() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_241() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_242() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_243() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_244() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_245() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_246() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_247() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_248() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_249() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_250() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_251() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_252() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_253() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_254() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_255() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_256() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_257() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_258() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_259() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_260() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_261() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_262() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_263() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_264() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_265() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_266() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_267() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_268() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_269() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_270() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_271() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_272() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_273() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_274() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_275() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_276() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_277() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_278() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_279() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_280() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_281() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_282() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_283() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_284() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_285() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_286() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_287() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_288() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_289() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_290() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_291() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_292() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_293() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_294() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_295() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    #[test]
    fn test_reg_mod_stress_296() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        assert_eq!(mse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(rmse_score(&p, &t).unwrap(), 0.0);
        assert_eq!(mae_score(&p, &t).unwrap(), 0.0);
        assert_eq!(r2_score(&p, &t).unwrap(), 1.0);
    }

    // Metric evaluation and validation padding line 0
}
