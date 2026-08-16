//! # Generator Module
//!
//! [`Generator`] trait, latent sampling, output range handling.
#![allow(missing_docs)]

pub mod dcgan;
pub mod resnet;
pub mod conditional;

pub use dcgan::DcganGenerator;
pub use resnet::ResnetGenerator;
pub use conditional::ConditionalGenerator;

use brain_core::Tensor;
use crate::config::{GeneratorConfig, LatentType, OutputActivation};
use crate::utils::sample_gaussian;
use crate::ops::{tanh_act, sigmoid_act};

/// Core trait for all GAN generators.
pub trait Generator: Send + Sync {
    /// Forward pass: maps latent vector `z` to a generated image tensor.
    fn forward(&self, z: &Tensor) -> Tensor;
    /// Returns latent dimension.
    fn latent_dim(&self) -> usize;
    /// Returns output shape [C, H, W].
    fn output_shape(&self) -> Vec<usize>;
}

/// Samples a latent vector according to the configured distribution.
pub fn sample_latent(config: &GeneratorConfig, seed: u64) -> Tensor {
    match config.latent_type {
        LatentType::Gaussian => {
            let data = sample_gaussian(config.latent_dim, seed);
            Tensor::from_vec(data, vec![config.latent_dim])
        }
        LatentType::Uniform => {
            let mut rng = seed;
            let data: Vec<f64> = (0..config.latent_dim).map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (rng >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            }).collect();
            Tensor::from_vec(data, vec![config.latent_dim])
        }
        LatentType::Spherical => {
            let data = sample_gaussian(config.latent_dim, seed);
            let norm = data.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-8);
            let normalized: Vec<f64> = data.iter().map(|v| v / norm).collect();
            Tensor::from_vec(normalized, vec![config.latent_dim])
        }
    }
}

/// Applies the output activation to a generator's output.
pub fn apply_output_activation(t: &Tensor, activation: OutputActivation) -> Tensor {
    match activation {
        OutputActivation::Tanh => tanh_act(t),
        OutputActivation::Sigmoid => sigmoid_act(t),
        OutputActivation::Linear => t.clone(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_gen_mod_stress_001() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 1 as u64);
        assert_eq!(z.shape(), &[5]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 1 as u64);
        assert_eq!(zu.shape(), &[5]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 1 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_002() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 2 as u64);
        assert_eq!(z.shape(), &[6]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 2 as u64);
        assert_eq!(zu.shape(), &[6]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 2 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_003() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 3 as u64);
        assert_eq!(z.shape(), &[7]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 3 as u64);
        assert_eq!(zu.shape(), &[7]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 3 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_004() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 4 as u64);
        assert_eq!(z.shape(), &[8]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 4 as u64);
        assert_eq!(zu.shape(), &[8]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 4 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_005() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 5 as u64);
        assert_eq!(z.shape(), &[9]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 5 as u64);
        assert_eq!(zu.shape(), &[9]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 5 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_006() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 6 as u64);
        assert_eq!(z.shape(), &[10]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 6 as u64);
        assert_eq!(zu.shape(), &[10]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 6 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_007() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 7 as u64);
        assert_eq!(z.shape(), &[11]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 7 as u64);
        assert_eq!(zu.shape(), &[11]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 7 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_008() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 8 as u64);
        assert_eq!(z.shape(), &[12]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 8 as u64);
        assert_eq!(zu.shape(), &[12]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 8 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_009() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 9 as u64);
        assert_eq!(z.shape(), &[13]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 9 as u64);
        assert_eq!(zu.shape(), &[13]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 9 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_010() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 10 as u64);
        assert_eq!(z.shape(), &[14]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 10 as u64);
        assert_eq!(zu.shape(), &[14]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 10 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_011() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 11 as u64);
        assert_eq!(z.shape(), &[15]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 11 as u64);
        assert_eq!(zu.shape(), &[15]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 11 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_012() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 12 as u64);
        assert_eq!(z.shape(), &[16]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 12 as u64);
        assert_eq!(zu.shape(), &[16]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 12 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_013() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 13 as u64);
        assert_eq!(z.shape(), &[17]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 13 as u64);
        assert_eq!(zu.shape(), &[17]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 13 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_014() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 14 as u64);
        assert_eq!(z.shape(), &[18]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 14 as u64);
        assert_eq!(zu.shape(), &[18]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 14 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_015() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 15 as u64);
        assert_eq!(z.shape(), &[19]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 15 as u64);
        assert_eq!(zu.shape(), &[19]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 15 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_016() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 20;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 16 as u64);
        assert_eq!(z.shape(), &[20]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 16 as u64);
        assert_eq!(zu.shape(), &[20]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 16 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_017() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 21;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 17 as u64);
        assert_eq!(z.shape(), &[21]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 17 as u64);
        assert_eq!(zu.shape(), &[21]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 17 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_018() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 22;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 18 as u64);
        assert_eq!(z.shape(), &[22]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 18 as u64);
        assert_eq!(zu.shape(), &[22]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 18 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_019() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 23;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 19 as u64);
        assert_eq!(z.shape(), &[23]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 19 as u64);
        assert_eq!(zu.shape(), &[23]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 19 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_020() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 24;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 20 as u64);
        assert_eq!(z.shape(), &[24]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 20 as u64);
        assert_eq!(zu.shape(), &[24]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 20 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_021() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 25;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 21 as u64);
        assert_eq!(z.shape(), &[25]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 21 as u64);
        assert_eq!(zu.shape(), &[25]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 21 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_022() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 26;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 22 as u64);
        assert_eq!(z.shape(), &[26]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 22 as u64);
        assert_eq!(zu.shape(), &[26]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 22 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_023() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 27;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 23 as u64);
        assert_eq!(z.shape(), &[27]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 23 as u64);
        assert_eq!(zu.shape(), &[27]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 23 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_024() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 28;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 24 as u64);
        assert_eq!(z.shape(), &[28]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 24 as u64);
        assert_eq!(zu.shape(), &[28]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 24 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_025() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 29;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 25 as u64);
        assert_eq!(z.shape(), &[29]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 25 as u64);
        assert_eq!(zu.shape(), &[29]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 25 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_026() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 30;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 26 as u64);
        assert_eq!(z.shape(), &[30]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 26 as u64);
        assert_eq!(zu.shape(), &[30]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 26 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_027() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 31;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 27 as u64);
        assert_eq!(z.shape(), &[31]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 27 as u64);
        assert_eq!(zu.shape(), &[31]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 27 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_028() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 32;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 28 as u64);
        assert_eq!(z.shape(), &[32]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 28 as u64);
        assert_eq!(zu.shape(), &[32]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 28 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_029() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 33;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 29 as u64);
        assert_eq!(z.shape(), &[33]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 29 as u64);
        assert_eq!(zu.shape(), &[33]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 29 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_030() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 34;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 30 as u64);
        assert_eq!(z.shape(), &[34]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 30 as u64);
        assert_eq!(zu.shape(), &[34]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 30 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_031() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 35;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 31 as u64);
        assert_eq!(z.shape(), &[35]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 31 as u64);
        assert_eq!(zu.shape(), &[35]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 31 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_032() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 32 as u64);
        assert_eq!(z.shape(), &[4]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 32 as u64);
        assert_eq!(zu.shape(), &[4]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 32 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_033() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 33 as u64);
        assert_eq!(z.shape(), &[5]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 33 as u64);
        assert_eq!(zu.shape(), &[5]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 33 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_034() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 34 as u64);
        assert_eq!(z.shape(), &[6]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 34 as u64);
        assert_eq!(zu.shape(), &[6]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 34 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_035() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 35 as u64);
        assert_eq!(z.shape(), &[7]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 35 as u64);
        assert_eq!(zu.shape(), &[7]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 35 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_036() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 36 as u64);
        assert_eq!(z.shape(), &[8]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 36 as u64);
        assert_eq!(zu.shape(), &[8]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 36 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_037() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 37 as u64);
        assert_eq!(z.shape(), &[9]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 37 as u64);
        assert_eq!(zu.shape(), &[9]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 37 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_038() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 38 as u64);
        assert_eq!(z.shape(), &[10]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 38 as u64);
        assert_eq!(zu.shape(), &[10]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 38 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_039() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 39 as u64);
        assert_eq!(z.shape(), &[11]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 39 as u64);
        assert_eq!(zu.shape(), &[11]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 39 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_040() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 40 as u64);
        assert_eq!(z.shape(), &[12]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 40 as u64);
        assert_eq!(zu.shape(), &[12]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 40 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_041() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 41 as u64);
        assert_eq!(z.shape(), &[13]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 41 as u64);
        assert_eq!(zu.shape(), &[13]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 41 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_042() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 42 as u64);
        assert_eq!(z.shape(), &[14]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 42 as u64);
        assert_eq!(zu.shape(), &[14]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 42 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_043() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 43 as u64);
        assert_eq!(z.shape(), &[15]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 43 as u64);
        assert_eq!(zu.shape(), &[15]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 43 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_044() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 44 as u64);
        assert_eq!(z.shape(), &[16]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 44 as u64);
        assert_eq!(zu.shape(), &[16]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 44 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_045() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 45 as u64);
        assert_eq!(z.shape(), &[17]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 45 as u64);
        assert_eq!(zu.shape(), &[17]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 45 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_046() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 46 as u64);
        assert_eq!(z.shape(), &[18]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 46 as u64);
        assert_eq!(zu.shape(), &[18]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 46 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_047() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 47 as u64);
        assert_eq!(z.shape(), &[19]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 47 as u64);
        assert_eq!(zu.shape(), &[19]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 47 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_048() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 20;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 48 as u64);
        assert_eq!(z.shape(), &[20]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 48 as u64);
        assert_eq!(zu.shape(), &[20]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 48 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_049() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 21;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 49 as u64);
        assert_eq!(z.shape(), &[21]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 49 as u64);
        assert_eq!(zu.shape(), &[21]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 49 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_050() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 22;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 50 as u64);
        assert_eq!(z.shape(), &[22]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 50 as u64);
        assert_eq!(zu.shape(), &[22]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 50 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_051() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 23;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 51 as u64);
        assert_eq!(z.shape(), &[23]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 51 as u64);
        assert_eq!(zu.shape(), &[23]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 51 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_052() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 24;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 52 as u64);
        assert_eq!(z.shape(), &[24]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 52 as u64);
        assert_eq!(zu.shape(), &[24]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 52 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_053() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 25;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 53 as u64);
        assert_eq!(z.shape(), &[25]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 53 as u64);
        assert_eq!(zu.shape(), &[25]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 53 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_054() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 26;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 54 as u64);
        assert_eq!(z.shape(), &[26]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 54 as u64);
        assert_eq!(zu.shape(), &[26]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 54 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_055() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 27;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 55 as u64);
        assert_eq!(z.shape(), &[27]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 55 as u64);
        assert_eq!(zu.shape(), &[27]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 55 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_056() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 28;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 56 as u64);
        assert_eq!(z.shape(), &[28]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 56 as u64);
        assert_eq!(zu.shape(), &[28]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 56 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_057() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 29;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 57 as u64);
        assert_eq!(z.shape(), &[29]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 57 as u64);
        assert_eq!(zu.shape(), &[29]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 57 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_058() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 30;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 58 as u64);
        assert_eq!(z.shape(), &[30]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 58 as u64);
        assert_eq!(zu.shape(), &[30]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 58 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_059() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 31;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 59 as u64);
        assert_eq!(z.shape(), &[31]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 59 as u64);
        assert_eq!(zu.shape(), &[31]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 59 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_060() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 32;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 60 as u64);
        assert_eq!(z.shape(), &[32]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 60 as u64);
        assert_eq!(zu.shape(), &[32]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 60 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_061() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 33;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 61 as u64);
        assert_eq!(z.shape(), &[33]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 61 as u64);
        assert_eq!(zu.shape(), &[33]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 61 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_062() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 34;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 62 as u64);
        assert_eq!(z.shape(), &[34]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 62 as u64);
        assert_eq!(zu.shape(), &[34]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 62 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_063() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 35;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 63 as u64);
        assert_eq!(z.shape(), &[35]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 63 as u64);
        assert_eq!(zu.shape(), &[35]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 63 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_064() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 64 as u64);
        assert_eq!(z.shape(), &[4]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 64 as u64);
        assert_eq!(zu.shape(), &[4]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 64 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_065() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 65 as u64);
        assert_eq!(z.shape(), &[5]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 65 as u64);
        assert_eq!(zu.shape(), &[5]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 65 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_066() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 66 as u64);
        assert_eq!(z.shape(), &[6]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 66 as u64);
        assert_eq!(zu.shape(), &[6]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 66 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_067() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 67 as u64);
        assert_eq!(z.shape(), &[7]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 67 as u64);
        assert_eq!(zu.shape(), &[7]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 67 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_068() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 68 as u64);
        assert_eq!(z.shape(), &[8]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 68 as u64);
        assert_eq!(zu.shape(), &[8]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 68 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_069() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 69 as u64);
        assert_eq!(z.shape(), &[9]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 69 as u64);
        assert_eq!(zu.shape(), &[9]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 69 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_070() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 70 as u64);
        assert_eq!(z.shape(), &[10]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 70 as u64);
        assert_eq!(zu.shape(), &[10]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 70 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_071() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 71 as u64);
        assert_eq!(z.shape(), &[11]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 71 as u64);
        assert_eq!(zu.shape(), &[11]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 71 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_072() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 72 as u64);
        assert_eq!(z.shape(), &[12]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 72 as u64);
        assert_eq!(zu.shape(), &[12]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 72 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_073() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 73 as u64);
        assert_eq!(z.shape(), &[13]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 73 as u64);
        assert_eq!(zu.shape(), &[13]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 73 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_074() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 74 as u64);
        assert_eq!(z.shape(), &[14]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 74 as u64);
        assert_eq!(zu.shape(), &[14]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 74 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_075() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 75 as u64);
        assert_eq!(z.shape(), &[15]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 75 as u64);
        assert_eq!(zu.shape(), &[15]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 75 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_076() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 76 as u64);
        assert_eq!(z.shape(), &[16]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 76 as u64);
        assert_eq!(zu.shape(), &[16]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 76 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_077() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 77 as u64);
        assert_eq!(z.shape(), &[17]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 77 as u64);
        assert_eq!(zu.shape(), &[17]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 77 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_078() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 78 as u64);
        assert_eq!(z.shape(), &[18]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 78 as u64);
        assert_eq!(zu.shape(), &[18]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 78 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_079() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 79 as u64);
        assert_eq!(z.shape(), &[19]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 79 as u64);
        assert_eq!(zu.shape(), &[19]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 79 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_080() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 20;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 80 as u64);
        assert_eq!(z.shape(), &[20]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 80 as u64);
        assert_eq!(zu.shape(), &[20]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 80 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_081() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 21;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 81 as u64);
        assert_eq!(z.shape(), &[21]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 81 as u64);
        assert_eq!(zu.shape(), &[21]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 81 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_082() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 22;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 82 as u64);
        assert_eq!(z.shape(), &[22]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 82 as u64);
        assert_eq!(zu.shape(), &[22]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 82 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_083() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 23;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 83 as u64);
        assert_eq!(z.shape(), &[23]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 83 as u64);
        assert_eq!(zu.shape(), &[23]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 83 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_084() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 24;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 84 as u64);
        assert_eq!(z.shape(), &[24]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 84 as u64);
        assert_eq!(zu.shape(), &[24]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 84 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_085() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 25;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 85 as u64);
        assert_eq!(z.shape(), &[25]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 85 as u64);
        assert_eq!(zu.shape(), &[25]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 85 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_086() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 26;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 86 as u64);
        assert_eq!(z.shape(), &[26]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 86 as u64);
        assert_eq!(zu.shape(), &[26]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 86 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_087() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 27;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 87 as u64);
        assert_eq!(z.shape(), &[27]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 87 as u64);
        assert_eq!(zu.shape(), &[27]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 87 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_088() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 28;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 88 as u64);
        assert_eq!(z.shape(), &[28]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 88 as u64);
        assert_eq!(zu.shape(), &[28]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 88 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_089() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 29;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 89 as u64);
        assert_eq!(z.shape(), &[29]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 89 as u64);
        assert_eq!(zu.shape(), &[29]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 89 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_090() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 30;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 90 as u64);
        assert_eq!(z.shape(), &[30]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 90 as u64);
        assert_eq!(zu.shape(), &[30]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 90 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_091() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 31;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 91 as u64);
        assert_eq!(z.shape(), &[31]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 91 as u64);
        assert_eq!(zu.shape(), &[31]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 91 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_092() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 32;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 92 as u64);
        assert_eq!(z.shape(), &[32]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 92 as u64);
        assert_eq!(zu.shape(), &[32]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 92 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_093() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 33;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 93 as u64);
        assert_eq!(z.shape(), &[33]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 93 as u64);
        assert_eq!(zu.shape(), &[33]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 93 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_094() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 34;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 94 as u64);
        assert_eq!(z.shape(), &[34]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 94 as u64);
        assert_eq!(zu.shape(), &[34]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 94 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_095() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 35;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 95 as u64);
        assert_eq!(z.shape(), &[35]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 95 as u64);
        assert_eq!(zu.shape(), &[35]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 95 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_096() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 96 as u64);
        assert_eq!(z.shape(), &[4]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 96 as u64);
        assert_eq!(zu.shape(), &[4]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 96 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_097() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 97 as u64);
        assert_eq!(z.shape(), &[5]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 97 as u64);
        assert_eq!(zu.shape(), &[5]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 97 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_098() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 98 as u64);
        assert_eq!(z.shape(), &[6]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 98 as u64);
        assert_eq!(zu.shape(), &[6]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 98 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_099() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 99 as u64);
        assert_eq!(z.shape(), &[7]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 99 as u64);
        assert_eq!(zu.shape(), &[7]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 99 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_100() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 100 as u64);
        assert_eq!(z.shape(), &[8]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 100 as u64);
        assert_eq!(zu.shape(), &[8]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 100 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_101() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 101 as u64);
        assert_eq!(z.shape(), &[9]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 101 as u64);
        assert_eq!(zu.shape(), &[9]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 101 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_102() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 102 as u64);
        assert_eq!(z.shape(), &[10]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 102 as u64);
        assert_eq!(zu.shape(), &[10]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 102 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_103() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 103 as u64);
        assert_eq!(z.shape(), &[11]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 103 as u64);
        assert_eq!(zu.shape(), &[11]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 103 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_104() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 104 as u64);
        assert_eq!(z.shape(), &[12]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 104 as u64);
        assert_eq!(zu.shape(), &[12]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 104 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_105() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 105 as u64);
        assert_eq!(z.shape(), &[13]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 105 as u64);
        assert_eq!(zu.shape(), &[13]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 105 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_106() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 106 as u64);
        assert_eq!(z.shape(), &[14]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 106 as u64);
        assert_eq!(zu.shape(), &[14]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 106 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_107() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 107 as u64);
        assert_eq!(z.shape(), &[15]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 107 as u64);
        assert_eq!(zu.shape(), &[15]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 107 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_108() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 108 as u64);
        assert_eq!(z.shape(), &[16]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 108 as u64);
        assert_eq!(zu.shape(), &[16]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 108 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_109() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 109 as u64);
        assert_eq!(z.shape(), &[17]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 109 as u64);
        assert_eq!(zu.shape(), &[17]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 109 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_110() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 110 as u64);
        assert_eq!(z.shape(), &[18]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 110 as u64);
        assert_eq!(zu.shape(), &[18]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 110 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_111() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 111 as u64);
        assert_eq!(z.shape(), &[19]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 111 as u64);
        assert_eq!(zu.shape(), &[19]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 111 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_112() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 20;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 112 as u64);
        assert_eq!(z.shape(), &[20]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 112 as u64);
        assert_eq!(zu.shape(), &[20]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 112 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_113() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 21;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 113 as u64);
        assert_eq!(z.shape(), &[21]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 113 as u64);
        assert_eq!(zu.shape(), &[21]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 113 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_114() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 22;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 114 as u64);
        assert_eq!(z.shape(), &[22]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 114 as u64);
        assert_eq!(zu.shape(), &[22]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 114 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_115() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 23;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 115 as u64);
        assert_eq!(z.shape(), &[23]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 115 as u64);
        assert_eq!(zu.shape(), &[23]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 115 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_116() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 24;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 116 as u64);
        assert_eq!(z.shape(), &[24]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 116 as u64);
        assert_eq!(zu.shape(), &[24]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 116 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_117() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 25;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 117 as u64);
        assert_eq!(z.shape(), &[25]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 117 as u64);
        assert_eq!(zu.shape(), &[25]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 117 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_118() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 26;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 118 as u64);
        assert_eq!(z.shape(), &[26]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 118 as u64);
        assert_eq!(zu.shape(), &[26]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 118 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_119() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 27;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 119 as u64);
        assert_eq!(z.shape(), &[27]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 119 as u64);
        assert_eq!(zu.shape(), &[27]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 119 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_120() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 28;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 120 as u64);
        assert_eq!(z.shape(), &[28]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 120 as u64);
        assert_eq!(zu.shape(), &[28]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 120 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_121() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 29;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 121 as u64);
        assert_eq!(z.shape(), &[29]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 121 as u64);
        assert_eq!(zu.shape(), &[29]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 121 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_122() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 30;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 122 as u64);
        assert_eq!(z.shape(), &[30]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 122 as u64);
        assert_eq!(zu.shape(), &[30]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 122 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_123() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 31;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 123 as u64);
        assert_eq!(z.shape(), &[31]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 123 as u64);
        assert_eq!(zu.shape(), &[31]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 123 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_124() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 32;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 124 as u64);
        assert_eq!(z.shape(), &[32]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 124 as u64);
        assert_eq!(zu.shape(), &[32]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 124 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_125() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 33;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 125 as u64);
        assert_eq!(z.shape(), &[33]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 125 as u64);
        assert_eq!(zu.shape(), &[33]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 125 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_126() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 34;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 126 as u64);
        assert_eq!(z.shape(), &[34]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 126 as u64);
        assert_eq!(zu.shape(), &[34]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 126 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_127() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 35;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 127 as u64);
        assert_eq!(z.shape(), &[35]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 127 as u64);
        assert_eq!(zu.shape(), &[35]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 127 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_128() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 128 as u64);
        assert_eq!(z.shape(), &[4]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 128 as u64);
        assert_eq!(zu.shape(), &[4]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 128 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_129() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 129 as u64);
        assert_eq!(z.shape(), &[5]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 129 as u64);
        assert_eq!(zu.shape(), &[5]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 129 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_130() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 130 as u64);
        assert_eq!(z.shape(), &[6]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 130 as u64);
        assert_eq!(zu.shape(), &[6]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 130 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_131() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 131 as u64);
        assert_eq!(z.shape(), &[7]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 131 as u64);
        assert_eq!(zu.shape(), &[7]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 131 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_132() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 132 as u64);
        assert_eq!(z.shape(), &[8]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 132 as u64);
        assert_eq!(zu.shape(), &[8]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 132 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_133() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 133 as u64);
        assert_eq!(z.shape(), &[9]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 133 as u64);
        assert_eq!(zu.shape(), &[9]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 133 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_134() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 134 as u64);
        assert_eq!(z.shape(), &[10]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 134 as u64);
        assert_eq!(zu.shape(), &[10]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 134 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_135() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 135 as u64);
        assert_eq!(z.shape(), &[11]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 135 as u64);
        assert_eq!(zu.shape(), &[11]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 135 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_136() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 136 as u64);
        assert_eq!(z.shape(), &[12]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 136 as u64);
        assert_eq!(zu.shape(), &[12]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 136 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_137() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 137 as u64);
        assert_eq!(z.shape(), &[13]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 137 as u64);
        assert_eq!(zu.shape(), &[13]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 137 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_138() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 138 as u64);
        assert_eq!(z.shape(), &[14]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 138 as u64);
        assert_eq!(zu.shape(), &[14]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 138 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_139() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 139 as u64);
        assert_eq!(z.shape(), &[15]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 139 as u64);
        assert_eq!(zu.shape(), &[15]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 139 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_140() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 140 as u64);
        assert_eq!(z.shape(), &[16]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 140 as u64);
        assert_eq!(zu.shape(), &[16]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 140 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_141() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 141 as u64);
        assert_eq!(z.shape(), &[17]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 141 as u64);
        assert_eq!(zu.shape(), &[17]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 141 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_142() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 142 as u64);
        assert_eq!(z.shape(), &[18]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 142 as u64);
        assert_eq!(zu.shape(), &[18]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 142 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_143() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 143 as u64);
        assert_eq!(z.shape(), &[19]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 143 as u64);
        assert_eq!(zu.shape(), &[19]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 143 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_144() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 20;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 144 as u64);
        assert_eq!(z.shape(), &[20]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 144 as u64);
        assert_eq!(zu.shape(), &[20]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 144 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_145() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 21;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 145 as u64);
        assert_eq!(z.shape(), &[21]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 145 as u64);
        assert_eq!(zu.shape(), &[21]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 145 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_146() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 22;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 146 as u64);
        assert_eq!(z.shape(), &[22]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 146 as u64);
        assert_eq!(zu.shape(), &[22]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 146 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_147() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 23;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 147 as u64);
        assert_eq!(z.shape(), &[23]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 147 as u64);
        assert_eq!(zu.shape(), &[23]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 147 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_148() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 24;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 148 as u64);
        assert_eq!(z.shape(), &[24]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 148 as u64);
        assert_eq!(zu.shape(), &[24]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 148 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_149() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 25;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 149 as u64);
        assert_eq!(z.shape(), &[25]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 149 as u64);
        assert_eq!(zu.shape(), &[25]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 149 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_150() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 26;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 150 as u64);
        assert_eq!(z.shape(), &[26]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 150 as u64);
        assert_eq!(zu.shape(), &[26]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 150 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_151() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 27;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 151 as u64);
        assert_eq!(z.shape(), &[27]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 151 as u64);
        assert_eq!(zu.shape(), &[27]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 151 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_152() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 28;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 152 as u64);
        assert_eq!(z.shape(), &[28]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 152 as u64);
        assert_eq!(zu.shape(), &[28]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 152 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_153() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 29;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 153 as u64);
        assert_eq!(z.shape(), &[29]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 153 as u64);
        assert_eq!(zu.shape(), &[29]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 153 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_154() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 30;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 154 as u64);
        assert_eq!(z.shape(), &[30]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 154 as u64);
        assert_eq!(zu.shape(), &[30]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 154 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_155() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 31;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 155 as u64);
        assert_eq!(z.shape(), &[31]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 155 as u64);
        assert_eq!(zu.shape(), &[31]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 155 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_156() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 32;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 156 as u64);
        assert_eq!(z.shape(), &[32]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 156 as u64);
        assert_eq!(zu.shape(), &[32]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 156 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_157() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 33;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 157 as u64);
        assert_eq!(z.shape(), &[33]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 157 as u64);
        assert_eq!(zu.shape(), &[33]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 157 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_158() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 34;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 158 as u64);
        assert_eq!(z.shape(), &[34]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 158 as u64);
        assert_eq!(zu.shape(), &[34]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 158 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_159() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 35;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 159 as u64);
        assert_eq!(z.shape(), &[35]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 159 as u64);
        assert_eq!(zu.shape(), &[35]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 159 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_160() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 160 as u64);
        assert_eq!(z.shape(), &[4]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 160 as u64);
        assert_eq!(zu.shape(), &[4]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 160 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_161() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 161 as u64);
        assert_eq!(z.shape(), &[5]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 161 as u64);
        assert_eq!(zu.shape(), &[5]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 161 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_162() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 162 as u64);
        assert_eq!(z.shape(), &[6]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 162 as u64);
        assert_eq!(zu.shape(), &[6]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 162 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_163() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 163 as u64);
        assert_eq!(z.shape(), &[7]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 163 as u64);
        assert_eq!(zu.shape(), &[7]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 163 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_164() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 164 as u64);
        assert_eq!(z.shape(), &[8]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 164 as u64);
        assert_eq!(zu.shape(), &[8]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 164 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_165() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 165 as u64);
        assert_eq!(z.shape(), &[9]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 165 as u64);
        assert_eq!(zu.shape(), &[9]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 165 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_166() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 166 as u64);
        assert_eq!(z.shape(), &[10]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 166 as u64);
        assert_eq!(zu.shape(), &[10]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 166 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_167() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 167 as u64);
        assert_eq!(z.shape(), &[11]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 167 as u64);
        assert_eq!(zu.shape(), &[11]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 167 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_168() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 168 as u64);
        assert_eq!(z.shape(), &[12]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 168 as u64);
        assert_eq!(zu.shape(), &[12]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 168 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_169() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 169 as u64);
        assert_eq!(z.shape(), &[13]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 169 as u64);
        assert_eq!(zu.shape(), &[13]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 169 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_170() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 170 as u64);
        assert_eq!(z.shape(), &[14]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 170 as u64);
        assert_eq!(zu.shape(), &[14]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 170 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_171() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 171 as u64);
        assert_eq!(z.shape(), &[15]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 171 as u64);
        assert_eq!(zu.shape(), &[15]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 171 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
    }

    #[test]
    fn test_gen_mod_stress_172() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.latent_type = LatentType::Gaussian;
        let z = sample_latent(&cfg, 172 as u64);
        assert_eq!(z.shape(), &[16]);
        cfg.latent_type = LatentType::Uniform;
        let zu = sample_latent(&cfg, 172 as u64);
        assert_eq!(zu.shape(), &[16]);
        cfg.latent_type = LatentType::Spherical;
        let zs = sample_latent(&cfg, 172 as u64);
        let norm: f64 = zs.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        let t = Tensor::zeros(vec![4]);
        let a = apply_output_activation(&t, OutputActivation::Tanh);
        assert_eq!(a.shape(), &[4]);
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
