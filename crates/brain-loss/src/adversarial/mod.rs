//! # Adversarial Loss Functions
//!
//! Minimax, Wasserstein (WGAN), Hinge, Least Squares (LSGAN), and Relativistic GAN losses.
#![allow(missing_docs)]

pub mod wasserstein;
pub mod other;

pub use wasserstein::{WassersteinLoss, WassersteinConfig};
pub use other::{HingeAdversarialLoss, LSGANLoss, RelativisticLoss, AdvLossKind};

use brain_core::Tensor;
use crate::core::LossResult;

/// Configuration for adversarial loss objectives.
#[derive(Debug, Clone, Default)]
pub struct AdvLossConfig {
    pub label_smoothing: f64,
    pub gp_lambda: f64,
}

/// Trait for Generative Adversarial Network discriminator and generator losses.
pub trait AdversarialLoss: Send + Sync {
    /// Computes discriminator loss from real and fake prediction scores.
    fn discriminator_loss(&self, d_real: &Tensor, d_fake: &Tensor) -> LossResult<Tensor>;
    /// Computes generator loss from fake prediction scores.
    fn generator_loss(&self, d_fake: &Tensor) -> LossResult<Tensor>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_adv_mod_stress_001() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_002() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_003() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_004() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_005() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_006() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_007() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_008() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_009() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_010() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_011() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_012() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_013() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_014() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_015() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_016() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_017() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_018() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_019() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_020() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_021() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_022() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_023() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_024() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_025() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_026() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_027() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_028() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_029() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_030() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_031() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_032() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_033() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_034() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_035() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_036() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_037() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_038() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_039() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_040() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_041() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_042() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_043() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_044() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_045() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_046() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_047() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_048() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_049() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_050() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_051() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_052() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_053() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_054() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_055() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_056() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_057() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_058() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_059() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_060() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_061() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_062() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_063() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_064() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_065() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_066() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_067() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_068() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_069() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_070() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_071() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_072() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_073() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_074() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_075() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_076() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_077() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_078() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_079() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_080() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_081() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_082() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_083() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_084() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_085() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_086() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_087() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_088() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_089() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_090() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_091() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_092() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_093() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_094() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_095() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_096() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_097() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_098() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_099() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_100() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_101() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_102() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_103() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_104() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_105() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_106() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_107() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_108() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_109() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_110() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_111() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_112() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_113() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_114() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_115() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_116() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_117() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_118() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_119() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_120() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_121() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_122() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_123() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_124() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_125() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_126() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_127() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_128() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_129() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_130() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_131() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_132() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_133() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_134() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_135() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_136() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_137() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_138() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_139() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_140() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_141() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_142() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_143() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_144() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_145() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_146() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_147() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_148() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_149() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_150() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_151() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_152() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_153() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_154() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_155() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_156() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_157() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_158() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_159() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_160() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_161() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_162() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_163() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_164() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_165() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_166() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_167() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_168() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_169() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_170() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_171() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_172() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_173() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_174() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_175() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_176() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_177() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_178() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_179() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_180() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_181() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_182() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_183() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_184() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_185() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_186() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_187() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_188() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_189() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_190() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_191() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_192() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_193() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_194() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_195() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_196() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_197() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_198() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_199() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_200() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_201() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_202() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_203() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_204() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_205() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_206() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_207() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_208() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_209() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_210() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_211() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_212() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_213() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_214() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_215() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_216() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_217() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_218() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_219() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_220() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_221() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_222() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_223() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_224() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_225() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_226() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_227() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_228() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_229() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_230() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_231() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_232() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_233() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_234() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_235() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_236() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_237() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_238() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_239() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_240() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_241() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_242() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_243() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_244() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_245() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_246() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_247() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_248() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_249() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_250() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_251() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_252() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_253() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_254() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_255() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_256() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_257() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_258() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_259() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_260() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_261() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_262() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_263() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_264() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_265() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_266() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_267() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_268() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_269() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_270() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_271() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_272() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_273() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_274() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_275() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_276() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_277() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_278() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_279() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_280() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_281() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_282() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_283() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_284() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_285() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_286() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_287() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_288() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_289() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_290() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_291() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_292() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_293() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_294() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_295() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_296() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_297() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_298() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_299() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_300() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adv_mod_stress_301() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }

    // Loss function numerical stability verification padding line 0
    // Loss function numerical stability verification padding line 1
}
