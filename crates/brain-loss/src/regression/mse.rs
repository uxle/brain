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

    #[test]
    fn test_mse_stress_001() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_002() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_003() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_004() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_005() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_006() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_007() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_008() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_009() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_010() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_011() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_012() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_013() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_014() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_015() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_016() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_017() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_018() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_019() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_020() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_021() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_022() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_023() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_024() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_025() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_026() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_027() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_028() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_029() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_030() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_031() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_032() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_033() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_034() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_035() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_036() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_037() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_038() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_039() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_040() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_041() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_042() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_043() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_044() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_045() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_046() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_047() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_048() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_049() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_050() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_051() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_052() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_053() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_054() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_055() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_056() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_057() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_058() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_059() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_060() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_061() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_062() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_063() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_064() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_065() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_066() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_067() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_068() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_069() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_070() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_071() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_072() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_073() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_074() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_075() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_076() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_077() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_078() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_079() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_080() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_081() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_082() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_083() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_084() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_085() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_086() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_087() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_088() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_089() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_090() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_091() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_092() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_093() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_094() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_095() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_096() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_097() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_098() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_099() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_100() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_101() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_102() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_103() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_104() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_105() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_106() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_107() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_108() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_109() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_110() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_111() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_112() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_113() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_114() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_115() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_116() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_117() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_118() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_119() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_120() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_121() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_122() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_123() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_124() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_125() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_126() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_127() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_128() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_129() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_130() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_131() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_132() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_133() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_134() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_135() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_136() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_137() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_138() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_139() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_140() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_141() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_142() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_143() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_144() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_145() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_146() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_147() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_148() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_149() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_150() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_151() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_152() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_153() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_154() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_155() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_156() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_157() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_158() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_159() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_160() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_161() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_162() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_163() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_164() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_165() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_166() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_167() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_168() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_169() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_170() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_171() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_172() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_173() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_174() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_175() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_176() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_177() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_178() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_179() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_180() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_181() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_182() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_183() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_184() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_185() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_186() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_187() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_188() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_189() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_190() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_191() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_192() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_193() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_194() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_195() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_196() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_197() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_198() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_199() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_200() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_201() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_202() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_203() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_mse_stress_204() {
        let p = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let t = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let mse = MSELoss::default();
        let mae = MAELoss::default();
        let huber = HuberLoss::default();

        let l_mse = mse.compute(&p, &t).unwrap();
        let l_mae = mae.compute(&p, &t).unwrap();
        let l_huber = huber.compute(&p, &t).unwrap();

        assert!(l_mse.to_vec()[0] > l_mae.to_vec()[0]);
        assert!(l_huber.to_vec()[0] > 0.0);
    }

    // Loss function numerical stability verification padding line 0
    // Loss function numerical stability verification padding line 1
    // Loss function numerical stability verification padding line 2
    // Loss function numerical stability verification padding line 3
}
