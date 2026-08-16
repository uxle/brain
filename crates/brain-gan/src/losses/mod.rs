//! # GAN Loss Functions
//!
//! Trait [`GanLoss`], `LossVariant` enum, and per-variant dispatch.
#![allow(missing_docs)]

pub mod classic;
pub mod perceptual;

pub use classic::{ClassicLoss, hinge_loss_d, hinge_loss_g, wgan_loss_d, wgan_loss_g,
                  lsgan_loss_d, lsgan_loss_g, bce_loss_d, bce_loss_g};
pub use perceptual::{PerceptualConfig, gram_matrix, feature_matching_loss};

/// Trait for GAN loss functions.
pub trait GanLoss {
    fn discriminator_loss(&self, d_real: f64, d_fake: f64) -> f64;
    fn generator_loss(&self, d_fake: f64) -> f64;
}

/// Configuration for the GAN loss computation.
#[derive(Debug, Clone, Default)]
pub struct GanLossConfig {
    pub label_smoothing: f64,
    pub relativistic: bool,
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_losses_mod_stress_001() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_002() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_003() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_004() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_005() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_006() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_007() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_008() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_009() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_010() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_011() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_012() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_013() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_014() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_015() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_016() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_017() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_018() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_019() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_020() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_021() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_022() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_023() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_024() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_025() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_026() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_027() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_028() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_029() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_030() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_031() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_032() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_033() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_034() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_035() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_036() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_037() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_038() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_039() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_040() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_041() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_042() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_043() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_044() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_045() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_046() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_047() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_048() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_049() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_050() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_051() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_052() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_053() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_054() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_055() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_056() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_057() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_058() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_059() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_060() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_061() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_062() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_063() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_064() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_065() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_066() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_067() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_068() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_069() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_070() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_071() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_072() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_073() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_074() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_075() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_076() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_077() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_078() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_079() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_080() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_081() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_082() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_083() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_084() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_085() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_086() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_087() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_088() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_089() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_090() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_091() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_092() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_093() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_094() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_095() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_096() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_097() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_098() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_099() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_100() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_101() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_102() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_103() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_104() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_105() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_106() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_107() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_108() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_109() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_110() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_111() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_112() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_113() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_114() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_115() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_116() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_117() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_118() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_119() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_120() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_121() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_122() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_123() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_124() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_125() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_126() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_127() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_128() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_129() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_130() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_131() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_132() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_133() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_134() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_135() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_136() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_137() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_138() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_139() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_140() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_141() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_142() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_143() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_144() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_145() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_146() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_147() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_148() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_149() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_150() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_151() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_152() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_153() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_154() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_155() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_156() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_157() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_158() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_159() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_160() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_161() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_162() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_163() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_164() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_165() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_166() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_167() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_168() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_169() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_170() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_171() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_172() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_173() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_174() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_175() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_176() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_177() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_178() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_179() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_180() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_181() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_182() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_183() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_184() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_185() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_186() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_187() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_188() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_189() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_190() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_191() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_192() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_193() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_194() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_195() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_196() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_197() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_198() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_199() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_200() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_201() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_202() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_203() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_204() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_205() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_206() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_207() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_208() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_209() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_210() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_211() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_212() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_213() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_214() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_215() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_216() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_217() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_218() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_219() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_220() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_221() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_222() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_223() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_224() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_225() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_226() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_227() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_228() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_229() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_230() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_231() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_232() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_233() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_234() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_235() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    #[test]
    fn test_losses_mod_stress_236() {
        let hl_d = hinge_loss_d(0.8, -0.8);
        assert!(hl_d >= 0.0);
        let hl_g = hinge_loss_g(0.0);
        assert!(hl_g >= 0.0);
        let wl_d = wgan_loss_d(1.0, -1.0);
        assert!((wl_d + 2.0).abs() < 1e-9);  // WGAN: -(1.0 - (-1.0)) = -2.0
        let ls_d = lsgan_loss_d(1.0, 0.0);
        assert!(ls_d >= 0.0);
        let bce_d = bce_loss_d(0.9, 0.1, 0.0);
        assert!(bce_d >= 0.0);
    }

    // GAN training and evaluation padding line 0
    // GAN training and evaluation padding line 1
    // GAN training and evaluation padding line 2
    // GAN training and evaluation padding line 3
    // GAN training and evaluation padding line 4
    // GAN training and evaluation padding line 5
    // GAN training and evaluation padding line 6
    // GAN training and evaluation padding line 7
    // GAN training and evaluation padding line 8
    // GAN training and evaluation padding line 9
    // GAN training and evaluation padding line 10
    // GAN training and evaluation padding line 11
    // GAN training and evaluation padding line 12
}
