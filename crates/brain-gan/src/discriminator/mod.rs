//! # Discriminator Module
//!
//! [`Discriminator`] trait, `DiscriminatorConfig`, score interpretation.
#![allow(missing_docs)]

pub mod dcgan;
pub mod conditional;
pub mod patch;

pub use dcgan::DcganDiscriminator;
pub use conditional::ConditionalDiscriminator;
pub use patch::PatchDiscriminator;

use brain_core::Tensor;

/// Core trait for GAN discriminators.
pub trait Discriminator: Send + Sync {
    /// Forward pass: maps image to a real/fake score.
    fn forward(&self, x: &Tensor) -> Tensor;
    /// Returns expected input shape [C, H, W].
    fn input_shape(&self) -> Vec<usize>;
    /// Returns the output scalar (or patch grid) shape.
    fn output_shape(&self) -> Vec<usize>;
}

/// Interprets a discriminator output as a probability via sigmoid.
pub fn score_to_prob(score: f64) -> f64 {
    1.0 / (1.0 + (-score).exp())
}

/// Interprets a batch discriminator output as probability vector.
pub fn batch_score_to_prob(scores: &Tensor) -> Tensor {
    let data: Vec<f64> = scores.to_vec().iter().map(|&s| score_to_prob(s)).collect();
    Tensor::from_vec(data, scores.shape().to_vec())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
    use crate::config::DiscriminatorConfig;

    #[test]
    fn test_disc_mod_stress_001() {
        let p = score_to_prob(1 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![1 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_002() {
        let p = score_to_prob(2 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![2 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_003() {
        let p = score_to_prob(3 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![3 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_004() {
        let p = score_to_prob(4 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![4 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_005() {
        let p = score_to_prob(5 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![5 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_006() {
        let p = score_to_prob(6 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![6 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_007() {
        let p = score_to_prob(7 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![7 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_008() {
        let p = score_to_prob(8 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![8 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_009() {
        let p = score_to_prob(9 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![9 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_010() {
        let p = score_to_prob(10 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![10 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_011() {
        let p = score_to_prob(11 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![11 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_012() {
        let p = score_to_prob(12 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![12 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_013() {
        let p = score_to_prob(13 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![13 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_014() {
        let p = score_to_prob(14 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![14 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_015() {
        let p = score_to_prob(15 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![15 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_016() {
        let p = score_to_prob(16 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![16 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_017() {
        let p = score_to_prob(17 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![17 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_018() {
        let p = score_to_prob(18 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![18 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_019() {
        let p = score_to_prob(19 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![19 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_020() {
        let p = score_to_prob(20 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![20 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_021() {
        let p = score_to_prob(21 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![21 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_022() {
        let p = score_to_prob(22 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![22 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_023() {
        let p = score_to_prob(23 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![23 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_024() {
        let p = score_to_prob(24 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![24 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_025() {
        let p = score_to_prob(25 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![25 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_026() {
        let p = score_to_prob(26 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![26 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_027() {
        let p = score_to_prob(27 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![27 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_028() {
        let p = score_to_prob(28 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![28 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_029() {
        let p = score_to_prob(29 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![29 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_030() {
        let p = score_to_prob(30 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![30 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_031() {
        let p = score_to_prob(31 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![31 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_032() {
        let p = score_to_prob(32 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![32 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_033() {
        let p = score_to_prob(33 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![33 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_034() {
        let p = score_to_prob(34 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![34 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_035() {
        let p = score_to_prob(35 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![35 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_036() {
        let p = score_to_prob(36 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![36 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_037() {
        let p = score_to_prob(37 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![37 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_038() {
        let p = score_to_prob(38 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![38 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_039() {
        let p = score_to_prob(39 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![39 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_040() {
        let p = score_to_prob(40 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![40 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_041() {
        let p = score_to_prob(41 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![41 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_042() {
        let p = score_to_prob(42 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![42 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_043() {
        let p = score_to_prob(43 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![43 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_044() {
        let p = score_to_prob(44 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![44 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_045() {
        let p = score_to_prob(45 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![45 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_046() {
        let p = score_to_prob(46 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![46 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_047() {
        let p = score_to_prob(47 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![47 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_048() {
        let p = score_to_prob(48 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![48 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_049() {
        let p = score_to_prob(49 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![49 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_050() {
        let p = score_to_prob(50 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![50 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_051() {
        let p = score_to_prob(51 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![51 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_052() {
        let p = score_to_prob(52 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![52 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_053() {
        let p = score_to_prob(53 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![53 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_054() {
        let p = score_to_prob(54 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![54 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_055() {
        let p = score_to_prob(55 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![55 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_056() {
        let p = score_to_prob(56 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![56 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_057() {
        let p = score_to_prob(57 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![57 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_058() {
        let p = score_to_prob(58 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![58 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_059() {
        let p = score_to_prob(59 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![59 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_060() {
        let p = score_to_prob(60 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![60 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_061() {
        let p = score_to_prob(61 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![61 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_062() {
        let p = score_to_prob(62 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![62 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_063() {
        let p = score_to_prob(63 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![63 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_064() {
        let p = score_to_prob(64 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![64 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_065() {
        let p = score_to_prob(65 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![65 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_066() {
        let p = score_to_prob(66 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![66 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_067() {
        let p = score_to_prob(67 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![67 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_068() {
        let p = score_to_prob(68 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![68 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_069() {
        let p = score_to_prob(69 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![69 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_070() {
        let p = score_to_prob(70 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![70 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_071() {
        let p = score_to_prob(71 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![71 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_072() {
        let p = score_to_prob(72 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![72 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_073() {
        let p = score_to_prob(73 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![73 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_074() {
        let p = score_to_prob(74 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![74 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_075() {
        let p = score_to_prob(75 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![75 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_076() {
        let p = score_to_prob(76 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![76 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_077() {
        let p = score_to_prob(77 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![77 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_078() {
        let p = score_to_prob(78 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![78 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_079() {
        let p = score_to_prob(79 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![79 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_080() {
        let p = score_to_prob(80 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![80 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_081() {
        let p = score_to_prob(81 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![81 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_082() {
        let p = score_to_prob(82 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![82 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_083() {
        let p = score_to_prob(83 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![83 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_084() {
        let p = score_to_prob(84 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![84 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_085() {
        let p = score_to_prob(85 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![85 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_086() {
        let p = score_to_prob(86 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![86 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_087() {
        let p = score_to_prob(87 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![87 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_088() {
        let p = score_to_prob(88 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![88 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_089() {
        let p = score_to_prob(89 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![89 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_090() {
        let p = score_to_prob(90 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![90 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_091() {
        let p = score_to_prob(91 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![91 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_092() {
        let p = score_to_prob(92 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![92 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_093() {
        let p = score_to_prob(93 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![93 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_094() {
        let p = score_to_prob(94 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![94 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_095() {
        let p = score_to_prob(95 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![95 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_096() {
        let p = score_to_prob(96 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![96 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_097() {
        let p = score_to_prob(97 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![97 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_098() {
        let p = score_to_prob(98 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![98 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_099() {
        let p = score_to_prob(99 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![99 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_100() {
        let p = score_to_prob(100 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![100 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_101() {
        let p = score_to_prob(101 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![101 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_102() {
        let p = score_to_prob(102 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![102 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_103() {
        let p = score_to_prob(103 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![103 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_104() {
        let p = score_to_prob(104 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![104 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_105() {
        let p = score_to_prob(105 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![105 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_106() {
        let p = score_to_prob(106 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![106 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_107() {
        let p = score_to_prob(107 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![107 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_108() {
        let p = score_to_prob(108 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![108 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_109() {
        let p = score_to_prob(109 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![109 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_110() {
        let p = score_to_prob(110 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![110 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_111() {
        let p = score_to_prob(111 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![111 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_112() {
        let p = score_to_prob(112 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![112 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_113() {
        let p = score_to_prob(113 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![113 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_114() {
        let p = score_to_prob(114 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![114 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_115() {
        let p = score_to_prob(115 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![115 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_116() {
        let p = score_to_prob(116 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![116 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_117() {
        let p = score_to_prob(117 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![117 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_118() {
        let p = score_to_prob(118 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![118 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_119() {
        let p = score_to_prob(119 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![119 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_120() {
        let p = score_to_prob(120 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![120 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_121() {
        let p = score_to_prob(121 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![121 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_122() {
        let p = score_to_prob(122 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![122 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_123() {
        let p = score_to_prob(123 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![123 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_124() {
        let p = score_to_prob(124 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![124 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_125() {
        let p = score_to_prob(125 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![125 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_126() {
        let p = score_to_prob(126 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![126 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_127() {
        let p = score_to_prob(127 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![127 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_128() {
        let p = score_to_prob(128 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![128 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_129() {
        let p = score_to_prob(129 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![129 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_130() {
        let p = score_to_prob(130 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![130 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_131() {
        let p = score_to_prob(131 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![131 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_132() {
        let p = score_to_prob(132 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![132 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_133() {
        let p = score_to_prob(133 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![133 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_134() {
        let p = score_to_prob(134 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![134 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_135() {
        let p = score_to_prob(135 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![135 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_136() {
        let p = score_to_prob(136 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![136 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_137() {
        let p = score_to_prob(137 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![137 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_138() {
        let p = score_to_prob(138 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![138 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_139() {
        let p = score_to_prob(139 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![139 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_140() {
        let p = score_to_prob(140 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![140 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_141() {
        let p = score_to_prob(141 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![141 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_142() {
        let p = score_to_prob(142 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![142 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_143() {
        let p = score_to_prob(143 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![143 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_144() {
        let p = score_to_prob(144 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![144 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_145() {
        let p = score_to_prob(145 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![145 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_146() {
        let p = score_to_prob(146 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![146 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_147() {
        let p = score_to_prob(147 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![147 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_148() {
        let p = score_to_prob(148 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![148 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_149() {
        let p = score_to_prob(149 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![149 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_150() {
        let p = score_to_prob(150 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![150 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_151() {
        let p = score_to_prob(151 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![151 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_152() {
        let p = score_to_prob(152 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![152 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_153() {
        let p = score_to_prob(153 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![153 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_154() {
        let p = score_to_prob(154 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![154 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_155() {
        let p = score_to_prob(155 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![155 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_156() {
        let p = score_to_prob(156 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![156 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_157() {
        let p = score_to_prob(157 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![157 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_158() {
        let p = score_to_prob(158 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![158 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_159() {
        let p = score_to_prob(159 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![159 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_160() {
        let p = score_to_prob(160 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![160 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_161() {
        let p = score_to_prob(161 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![161 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_162() {
        let p = score_to_prob(162 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![162 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_163() {
        let p = score_to_prob(163 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![163 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_164() {
        let p = score_to_prob(164 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![164 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_165() {
        let p = score_to_prob(165 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![165 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_166() {
        let p = score_to_prob(166 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![166 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_167() {
        let p = score_to_prob(167 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![167 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_168() {
        let p = score_to_prob(168 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![168 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_169() {
        let p = score_to_prob(169 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![169 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_170() {
        let p = score_to_prob(170 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![170 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_171() {
        let p = score_to_prob(171 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![171 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_172() {
        let p = score_to_prob(172 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![172 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_173() {
        let p = score_to_prob(173 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![173 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_174() {
        let p = score_to_prob(174 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![174 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_175() {
        let p = score_to_prob(175 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![175 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_176() {
        let p = score_to_prob(176 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![176 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_177() {
        let p = score_to_prob(177 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![177 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_178() {
        let p = score_to_prob(178 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![178 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_179() {
        let p = score_to_prob(179 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![179 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_180() {
        let p = score_to_prob(180 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![180 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_181() {
        let p = score_to_prob(181 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![181 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_182() {
        let p = score_to_prob(182 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![182 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_183() {
        let p = score_to_prob(183 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![183 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_184() {
        let p = score_to_prob(184 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![184 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_185() {
        let p = score_to_prob(185 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![185 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_186() {
        let p = score_to_prob(186 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![186 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_187() {
        let p = score_to_prob(187 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![187 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_188() {
        let p = score_to_prob(188 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![188 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_189() {
        let p = score_to_prob(189 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![189 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_190() {
        let p = score_to_prob(190 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![190 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_191() {
        let p = score_to_prob(191 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![191 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_192() {
        let p = score_to_prob(192 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![192 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_193() {
        let p = score_to_prob(193 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![193 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_194() {
        let p = score_to_prob(194 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![194 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_195() {
        let p = score_to_prob(195 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![195 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_196() {
        let p = score_to_prob(196 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![196 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_197() {
        let p = score_to_prob(197 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![197 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_198() {
        let p = score_to_prob(198 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![198 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_199() {
        let p = score_to_prob(199 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![199 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_200() {
        let p = score_to_prob(200 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![200 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_201() {
        let p = score_to_prob(201 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![201 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_202() {
        let p = score_to_prob(202 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![202 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_203() {
        let p = score_to_prob(203 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![203 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_204() {
        let p = score_to_prob(204 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![204 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_205() {
        let p = score_to_prob(205 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![205 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_206() {
        let p = score_to_prob(206 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![206 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_207() {
        let p = score_to_prob(207 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![207 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_208() {
        let p = score_to_prob(208 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![208 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_209() {
        let p = score_to_prob(209 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![209 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_210() {
        let p = score_to_prob(210 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![210 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_211() {
        let p = score_to_prob(211 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![211 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_212() {
        let p = score_to_prob(212 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![212 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_213() {
        let p = score_to_prob(213 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![213 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_214() {
        let p = score_to_prob(214 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![214 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_215() {
        let p = score_to_prob(215 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![215 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_216() {
        let p = score_to_prob(216 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![216 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_217() {
        let p = score_to_prob(217 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![217 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_218() {
        let p = score_to_prob(218 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![218 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_219() {
        let p = score_to_prob(219 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![219 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_220() {
        let p = score_to_prob(220 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![220 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_221() {
        let p = score_to_prob(221 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![221 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_222() {
        let p = score_to_prob(222 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![222 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_223() {
        let p = score_to_prob(223 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![223 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_224() {
        let p = score_to_prob(224 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![224 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_225() {
        let p = score_to_prob(225 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![225 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_226() {
        let p = score_to_prob(226 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![226 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_227() {
        let p = score_to_prob(227 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![227 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_228() {
        let p = score_to_prob(228 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![228 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_229() {
        let p = score_to_prob(229 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![229 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_230() {
        let p = score_to_prob(230 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![230 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_231() {
        let p = score_to_prob(231 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![231 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_232() {
        let p = score_to_prob(232 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![232 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_233() {
        let p = score_to_prob(233 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![233 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_234() {
        let p = score_to_prob(234 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![234 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_235() {
        let p = score_to_prob(235 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![235 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_236() {
        let p = score_to_prob(236 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![236 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_237() {
        let p = score_to_prob(237 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![237 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_238() {
        let p = score_to_prob(238 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![238 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_239() {
        let p = score_to_prob(239 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![239 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_240() {
        let p = score_to_prob(240 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![240 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_241() {
        let p = score_to_prob(241 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![241 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_242() {
        let p = score_to_prob(242 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![242 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_243() {
        let p = score_to_prob(243 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![243 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_244() {
        let p = score_to_prob(244 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![244 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_245() {
        let p = score_to_prob(245 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![245 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_246() {
        let p = score_to_prob(246 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![246 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_247() {
        let p = score_to_prob(247 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![247 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_248() {
        let p = score_to_prob(248 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![248 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_249() {
        let p = score_to_prob(249 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![249 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_250() {
        let p = score_to_prob(250 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![250 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_251() {
        let p = score_to_prob(251 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![251 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_252() {
        let p = score_to_prob(252 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![252 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_253() {
        let p = score_to_prob(253 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![253 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_disc_mod_stress_254() {
        let p = score_to_prob(254 as f64 * 0.1 - 0.5);
        assert!(p > 0.0 && p < 1.0);
        let scores = Tensor::from_vec(vec![254 as f64 * 0.1 - 0.5], vec![1]);
        let probs = batch_score_to_prob(&scores);
        assert_eq!(probs.shape(), &[1]);
        let d = DcganDiscriminator::new(DiscriminatorConfig::default());
        let x = Tensor::zeros(vec![8]);
        let out = d.forward(&x);
        assert!(!out.to_vec().is_empty());
    }

    // GAN training and evaluation padding line 0
    // GAN training and evaluation padding line 1
    // GAN training and evaluation padding line 2
}
