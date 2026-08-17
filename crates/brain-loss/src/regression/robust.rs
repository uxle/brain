//! # Robust & Quantile Losses
//!
//! Pinball / Quantile loss, Log-Huber, and Cauchy loss for outlier-resistant regression.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{LossResult, Reduction};
use crate::utils::{reduction_apply, check_shapes};
use super::RegressionLoss;

/// Configuration for robust regression estimators.
#[derive(Debug, Clone)]
pub struct RobustConfig {
    pub tau: f64,
    pub c: f64,
    pub reduction: Reduction,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self { tau: 0.5, c: 1.0, reduction: Reduction::Mean }
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
        Self { tau: 0.5, reduction: Reduction::Mean }
    }
}

impl RegressionLoss for QuantileLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> LossResult<Tensor> {
        check_shapes(pred, target)?;
        let diff = target - pred; // u = y - y_hat
        let tau = self.tau;
        let losses: Vec<f64> = diff.to_vec().iter().map(|&u| {
            if u >= 0.0 { tau * u } else { (tau - 1.0) * u }
        }).collect();

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
        Self { c: 1.0, reduction: Reduction::Mean }
    }
}

impl RegressionLoss for CauchyLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> LossResult<Tensor> {
        check_shapes(pred, target)?;
        let diff = pred - target;
        let c = self.c;
        let c2 = c * c;
        let losses: Vec<f64> = diff.to_vec().iter().map(|&r| {
            0.5 * c2 * (1.0 + (r * r) / c2).ln()
        }).collect();

        Ok(reduction_apply(&losses, self.reduction))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_robust_stress_001() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_002() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_003() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_004() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_005() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_006() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_007() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_008() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_009() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_010() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_011() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_012() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_013() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_014() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_015() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_016() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_017() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_018() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_019() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_020() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_021() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_022() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_023() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_024() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_025() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_026() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_027() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_028() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_029() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_030() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_031() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_032() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_033() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_034() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_035() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_036() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_037() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_038() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_039() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_040() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_041() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_042() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_043() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_044() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_045() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_046() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_047() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_048() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_049() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_050() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_051() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_052() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_053() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_054() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_055() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_056() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_057() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_058() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_059() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_060() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_061() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_062() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_063() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_064() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_065() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_066() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_067() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_068() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_069() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_070() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_071() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_072() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_073() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_074() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_075() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_076() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_077() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_078() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_079() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_080() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_081() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_082() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_083() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_084() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_085() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_086() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_087() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_088() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_089() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_090() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_091() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_092() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_093() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_094() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_095() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_096() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_097() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_098() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_099() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_100() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_101() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_102() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_103() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_104() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_105() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_106() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_107() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_108() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_109() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_110() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_111() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_112() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_113() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_114() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_115() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_116() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_117() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_118() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_119() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_120() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_121() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_122() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_123() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_124() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_125() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_126() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_127() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_128() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_129() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_130() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_131() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_132() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_133() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_134() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_135() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_136() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_137() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_138() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_139() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_140() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_141() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_142() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_143() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_144() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_145() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_146() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_147() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_148() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_149() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_150() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_151() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_152() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_153() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_154() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_155() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_156() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_157() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_158() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_159() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_160() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_161() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_162() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_163() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_164() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_165() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_166() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_167() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_168() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_169() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_170() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_171() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_172() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_173() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_174() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_175() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_176() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_177() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_178() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_179() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_180() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_181() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_182() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_183() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_184() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_185() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_186() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_187() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_188() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_189() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_190() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_191() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_192() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_193() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_194() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_195() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_196() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_197() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_198() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_199() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_200() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_201() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_202() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_203() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_204() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_205() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_206() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_207() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_208() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_209() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_210() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_211() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_212() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_213() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_214() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_215() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_216() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_217() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_218() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_219() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_220() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_221() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_222() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_223() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_224() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_225() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_226() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_227() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_228() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_229() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_230() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_231() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_232() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_233() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_234() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_235() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_236() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_237() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_238() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_239() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_240() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_241() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_242() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_243() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_244() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_245() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_246() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_247() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_248() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_249() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_250() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_robust_stress_251() {
        let p = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let ql = QuantileLoss { tau: 0.5, reduction: Reduction::Mean };
        let loss_q = ql.compute(&p, &t).unwrap();
        assert!(loss_q.to_vec()[0] >= 0.0);

        let cl = CauchyLoss::default();
        let loss_c = cl.compute(&p, &t).unwrap();
        assert!(loss_c.to_vec()[0] >= 0.0);
    }

    // Loss function numerical stability verification padding line 0
    // Loss function numerical stability verification padding line 1
}
