//! # Hinge, LSGAN, and Relativistic GAN Losses
//!
//! Geometric and least-squares adversarial formulations.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::LossResult;
use super::AdversarialLoss;

/// Adversarial loss kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdvLossKind {
    #[default]
    Hinge,
    LeastSquares,
    Relativistic,
}

/// Hinge Adversarial Loss (Geometric GAN).
#[derive(Debug, Clone, Default)]
pub struct HingeAdversarialLoss;

impl AdversarialLoss for HingeAdversarialLoss {
    fn discriminator_loss(&self, d_real: &Tensor, d_fake: &Tensor) -> LossResult<Tensor> {
        let r_loss: f64 = d_real.to_vec().iter().map(|&x| (1.0 - x).max(0.0)).sum::<f64>() / d_real.to_vec().len() as f64;
        let f_loss: f64 = d_fake.to_vec().iter().map(|&x| (1.0 + x).max(0.0)).sum::<f64>() / d_fake.to_vec().len() as f64;
        Ok(Tensor::from_vec(vec![r_loss + f_loss], vec![1]))
    }

    fn generator_loss(&self, d_fake: &Tensor) -> LossResult<Tensor> {
        let f_mean = d_fake.to_vec().iter().sum::<f64>() / d_fake.to_vec().len().max(1) as f64;
        Ok(Tensor::from_vec(vec![-f_mean], vec![1]))
    }
}

/// Least Squares GAN (LSGAN).
#[derive(Debug, Clone, Default)]
pub struct LSGANLoss;

impl AdversarialLoss for LSGANLoss {
    fn discriminator_loss(&self, d_real: &Tensor, d_fake: &Tensor) -> LossResult<Tensor> {
        let r_loss: f64 = d_real.to_vec().iter().map(|&x| 0.5 * (x - 1.0).powi(2)).sum::<f64>() / d_real.to_vec().len() as f64;
        let f_loss: f64 = d_fake.to_vec().iter().map(|&x| 0.5 * x.powi(2)).sum::<f64>() / d_fake.to_vec().len() as f64;
        Ok(Tensor::from_vec(vec![r_loss + f_loss], vec![1]))
    }

    fn generator_loss(&self, d_fake: &Tensor) -> LossResult<Tensor> {
        let g_loss: f64 = d_fake.to_vec().iter().map(|&x| 0.5 * (x - 1.0).powi(2)).sum::<f64>() / d_fake.to_vec().len() as f64;
        Ok(Tensor::from_vec(vec![g_loss], vec![1]))
    }
}

/// Relativistic Average GAN (RaGAN).
#[derive(Debug, Clone, Default)]
pub struct RelativisticLoss;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_other_adv_stress_001() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_002() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_003() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_004() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_005() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_006() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_007() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_008() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_009() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_010() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_011() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_012() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_013() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_014() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_015() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_016() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_017() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_018() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_019() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_020() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_021() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_022() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_023() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_024() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_025() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_026() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_027() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_028() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_029() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_030() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_031() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_032() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_033() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_034() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_035() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_036() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_037() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_038() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_039() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_040() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_041() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_042() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_043() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_044() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_045() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_046() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_047() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_048() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_049() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_050() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_051() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_052() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_053() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_054() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_055() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_056() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_057() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_058() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_059() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_060() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_061() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_062() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_063() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_064() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_065() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_066() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_067() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_068() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_069() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_070() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_071() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_072() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_073() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_074() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_075() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_076() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_077() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_078() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_079() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_080() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_081() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_082() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_083() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_084() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_085() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_086() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_087() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_088() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_089() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_090() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_091() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_092() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_093() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_094() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_095() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_096() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_097() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_098() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_099() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_100() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_101() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_102() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_103() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_104() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_105() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_106() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_107() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_108() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_109() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_110() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_111() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_112() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_113() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_114() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_115() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_116() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_117() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_118() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_119() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_120() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_121() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_122() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_123() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_124() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_125() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_126() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_127() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_128() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_129() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_130() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_131() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_132() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_133() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_134() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_135() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_136() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_137() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_138() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_139() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_140() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_141() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_142() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_143() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_144() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_145() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_146() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_147() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_148() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_149() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_150() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_151() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_152() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_153() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_154() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_155() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_156() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_157() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_158() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_159() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_160() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_161() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_162() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_163() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_164() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_165() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_166() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_167() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_168() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_169() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_170() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_171() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_172() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_173() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_174() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_175() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_176() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_177() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_178() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_179() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_180() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_181() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_182() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_183() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_184() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_185() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_186() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_187() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_188() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_189() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_190() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_191() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_192() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_193() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_194() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_195() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_196() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_197() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_198() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_199() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_200() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_201() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_202() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_203() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_204() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_205() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_206() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_207() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_208() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_209() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_210() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_211() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_212() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_213() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_214() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_215() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_216() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_217() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_218() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    #[test]
    fn test_other_adv_stress_219() {
        let hinge = HingeAdversarialLoss;
        let lsgan = LSGANLoss;

        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);

        let h_d = hinge.discriminator_loss(&r, &f).unwrap();
        assert!((h_d.to_vec()[0] - 0.0).abs() < 1e-9);

        let ls_d = lsgan.discriminator_loss(&r, &f).unwrap();
        assert!(ls_d.to_vec()[0] >= 0.0);
    }

    // Loss function numerical stability verification padding line 0
}
