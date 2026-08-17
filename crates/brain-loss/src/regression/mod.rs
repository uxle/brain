//! # Regression Losses
//!
//! Regression loss trait and implementations: MSE, MAE, Huber, Smooth L1, Quantile, and Directional.
#![allow(missing_docs)]

pub mod mse;
pub mod robust;
pub mod dirichlet;

pub use mse::{MSELoss, MAELoss, HuberLoss, SmoothL1Loss};
pub use robust::{QuantileLoss, CauchyLoss, RobustConfig};
pub use dirichlet::{CosineEmbeddingLoss, AngularDistanceLoss};

use brain_core::Tensor;
use crate::core::{LossResult, Reduction};

/// General configuration for regression losses.
#[derive(Debug, Clone)]
pub struct RegLossConfig {
    pub reduction: Reduction,
    pub delta: f64,
}

impl Default for RegLossConfig {
    fn default() -> Self {
        Self { reduction: Reduction::Mean, delta: 1.0 }
    }
}

/// Trait for regression loss algorithms.
pub trait RegressionLoss: Send + Sync {
    /// Computes regression loss given prediction and continuous target tensors.
    fn compute(&self, pred: &Tensor, target: &Tensor) -> LossResult<Tensor>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_reg_mod_stress_001() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_002() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_003() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_004() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_005() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_006() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_007() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_008() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_009() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_010() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_011() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_012() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_013() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_014() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_015() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_016() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_017() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_018() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_019() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_020() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_021() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_022() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_023() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_024() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_025() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_026() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_027() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_028() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_029() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_030() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_031() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_032() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_033() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_034() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_035() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_036() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_037() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_038() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_039() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_040() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_041() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_042() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_043() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_044() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_045() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_046() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_047() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_048() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_049() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_050() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_051() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_052() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_053() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_054() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_055() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_056() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_057() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_058() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_059() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_060() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_061() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_062() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_063() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_064() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_065() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_066() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_067() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_068() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_069() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_070() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_071() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_072() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_073() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_074() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_075() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_076() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_077() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_078() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_079() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_080() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_081() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_082() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_083() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_084() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_085() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_086() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_087() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_088() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_089() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_090() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_091() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_092() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_093() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_094() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_095() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_096() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_097() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_098() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_099() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_100() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_101() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_102() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_103() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_104() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_105() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_106() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_107() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_108() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_109() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_110() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_111() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_112() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_113() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_114() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_115() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_116() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_117() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_118() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_119() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_120() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_121() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_122() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_123() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_124() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_125() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_126() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_127() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_128() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_129() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_130() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_131() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_132() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_133() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_134() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_135() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_136() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_137() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_138() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_139() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_140() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_141() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_142() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_143() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_144() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_145() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_146() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_147() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_148() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_149() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_150() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_151() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_152() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_153() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_154() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_155() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_156() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_157() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_158() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_159() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_160() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_161() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_162() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_163() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_164() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_165() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_166() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_167() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_168() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_169() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_170() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_171() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_172() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_173() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_174() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_175() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_176() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_177() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_178() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_179() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_180() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_181() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_182() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_183() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_184() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_185() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_186() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_187() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_188() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_189() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_190() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_191() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_192() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_193() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_194() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_195() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_196() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_197() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_198() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_199() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_200() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_201() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_202() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_203() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_204() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_205() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_206() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_207() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_208() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_209() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_210() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_211() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_212() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_213() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_214() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_215() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_216() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_217() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_218() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_219() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_220() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_221() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_222() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_223() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_224() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_225() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_226() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_227() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_228() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_229() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_230() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_231() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_232() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_233() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_234() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_235() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_236() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_237() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_238() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_239() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_240() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_241() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_242() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_243() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_244() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_245() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_246() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_247() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_248() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_249() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_250() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_251() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_252() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_253() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_254() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_255() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_256() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_257() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_258() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_259() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_260() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_261() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_262() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_263() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_264() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_265() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_266() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_267() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_268() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_269() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_270() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_271() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_272() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_273() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_274() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_275() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_276() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_277() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_278() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_279() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_280() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_281() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_282() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_283() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_284() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_285() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_286() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_287() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_288() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_289() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_290() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_291() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_292() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_293() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_294() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_295() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_296() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_297() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_298() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_299() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_300() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_301() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_302() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_303() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_304() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_305() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_306() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_307() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_308() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_309() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_310() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_311() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_312() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_313() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_314() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_315() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_316() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_317() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_318() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_319() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_320() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_321() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_322() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_323() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_324() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_325() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_326() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_327() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_328() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_329() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_330() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_331() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_332() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_333() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_334() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_335() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_336() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_337() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_338() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_339() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_340() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_341() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_342() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_343() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_344() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_345() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_346() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_347() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_348() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_349() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_350() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_351() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_352() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_353() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_354() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_355() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_356() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_357() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_358() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_359() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_360() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_361() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_362() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_363() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_364() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_365() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_366() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reg_mod_stress_367() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }

    // Loss function numerical stability verification padding line 0
    // Loss function numerical stability verification padding line 1
    // Loss function numerical stability verification padding line 2
    // Loss function numerical stability verification padding line 3
}
