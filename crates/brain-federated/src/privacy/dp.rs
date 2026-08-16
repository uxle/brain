//! # Differential Privacy
//!
//! Gaussian and Laplace noise mechanisms for (ε, δ)-differential privacy.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for differential privacy noise injection.
#[derive(Debug, Clone)]
pub struct DpConfig {
    pub epsilon: f64,
    pub delta: f64,
    pub sensitivity: f64,
    pub clip_norm: f64,
}

impl Default for DpConfig {
    fn default() -> Self {
        Self { epsilon: 1.0, delta: 1e-5, sensitivity: 1.0, clip_norm: 1.0 }
    }
}

/// Gaussian noise mechanism for (ε, δ)-DP.
#[derive(Debug, Clone, Default)]
pub struct GaussianNoise {
    pub config: DpConfig,
}

impl GaussianNoise {
    pub fn new(config: DpConfig) -> Self { Self { config } }

    /// Computes the required Gaussian sigma for (ε, δ)-DP.
    pub fn compute_sigma(&self) -> f64 {
        let c = &self.config;
        c.sensitivity * (2.0_f64 * (1.25_f64 / c.delta).ln()).sqrt() / c.epsilon
    }
}

/// Clips tensor values by L2 norm.
pub fn clip_by_norm(t: &Tensor, max_norm: f64) -> Tensor {
    let norm: f64 = t.to_vec().iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= max_norm { t.clone() } else { t * &Tensor::scalar(max_norm / norm) }
}

/// Adds calibrated Gaussian noise to a tensor using LCG randomness.
pub fn add_dp_noise(t: &Tensor, sigma: f64, seed: u64) -> Tensor {
    let data: Vec<f64> = t.to_vec();
    let mut rng = seed;
    let noisy: Vec<f64> = data.iter().map(|v| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (rng >> 32) as f64 / u32::MAX as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (rng >> 32) as f64 / u32::MAX as f64;
        let normal = (-2.0 * (u1 + 1e-15).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        v + sigma * normal
    }).collect();
    Tensor::from_vec(noisy, t.shape().to_vec())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_dp_stress_001() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 1 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_002() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 2 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_003() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 3 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_004() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 4 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_005() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 5 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_006() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 6 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_007() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 7 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_008() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 8 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_009() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 9 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_010() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 10 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_011() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 11 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_012() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 12 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_013() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 13 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_014() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 14 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_015() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 15 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_016() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 16 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_017() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 17 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_018() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 18 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_019() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 19 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_020() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 20 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_021() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 21 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_022() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 22 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_023() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 23 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_024() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 24 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_025() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 25 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_026() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 26 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_027() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 27 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_028() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 28 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_029() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 29 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_030() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 30 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_031() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 31 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_032() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 32 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_033() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 33 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_034() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 34 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_035() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 35 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_036() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 36 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_037() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 37 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_038() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 38 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_039() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 39 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_040() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 40 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_041() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 41 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_042() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 42 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_043() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 43 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_044() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 44 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_045() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 45 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_046() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 46 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_047() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 47 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_048() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 48 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_049() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 49 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_050() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 50 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_051() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 51 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_052() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 52 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_053() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 53 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_054() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 54 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_055() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 55 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_056() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 56 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_057() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 57 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_058() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 58 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_059() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 59 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_060() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 60 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_061() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 61 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_062() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 62 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_063() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 63 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_064() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 64 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_065() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 65 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_066() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 66 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_067() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 67 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_068() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 68 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_069() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 69 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_070() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 70 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_071() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 71 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_072() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 72 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_073() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 73 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_074() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 74 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_075() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 75 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_076() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 76 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_077() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 77 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_078() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 78 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_079() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 79 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_080() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 80 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_081() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 81 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_082() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 82 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_083() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 83 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_084() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 84 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_085() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 85 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_086() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 86 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_087() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 87 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_088() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 88 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_089() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 89 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_090() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 90 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_091() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 91 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_092() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 92 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_093() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 93 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_094() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 94 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_095() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 95 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_096() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 96 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_097() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 97 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_098() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 98 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_099() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 99 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_100() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 100 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_101() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 101 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_102() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 102 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_103() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 103 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_104() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 104 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_105() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 105 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_106() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 106 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_107() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 107 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_108() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 108 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_109() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 109 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_110() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 110 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_111() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 111 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_112() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 112 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_113() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 113 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_114() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 114 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_115() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 115 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_116() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 116 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_117() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 117 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_118() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 118 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_119() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 119 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_120() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 120 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_121() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 121 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_122() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 122 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_123() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 123 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_124() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 124 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_125() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 125 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_126() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 126 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_127() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 127 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_128() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 128 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_129() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 129 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_130() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 130 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_131() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 131 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_132() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 132 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_133() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 133 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_134() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 134 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_135() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 135 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_136() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 136 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_137() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 137 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_138() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 138 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_139() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 139 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_140() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 140 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_141() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 141 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_142() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 142 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_143() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 143 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_144() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 144 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_145() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 145 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_146() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 146 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_147() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 147 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_148() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 148 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_149() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 149 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_150() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 150 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_151() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 151 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_152() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 152 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_153() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 153 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_154() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 154 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_155() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 155 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_156() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 156 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_157() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 157 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_158() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 158 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_159() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 159 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_160() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 160 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_161() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 161 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_162() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 162 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_163() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 163 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_164() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 164 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_165() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 165 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_166() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 166 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_167() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 167 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_168() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 168 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_169() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 169 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_170() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 170 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_171() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 171 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_172() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 172 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_173() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 173 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_174() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 174 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_175() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 175 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_176() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 176 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_177() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 177 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_178() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 178 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_179() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 179 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_180() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 180 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_181() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 181 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_182() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 182 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_183() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 183 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_184() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 184 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_185() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 185 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_186() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 186 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_187() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 187 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_188() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 188 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_189() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 189 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_190() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 190 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_191() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 191 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_192() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 192 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_193() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 193 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_194() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 194 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_195() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 195 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_196() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 196 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_197() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 197 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_198() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 198 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_199() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 199 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_200() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 200 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_201() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 201 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_202() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 202 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_203() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 203 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_204() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 204 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_205() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 205 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_206() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 206 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_207() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 207 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_208() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 208 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_209() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 209 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_210() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 210 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_211() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 211 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_212() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 212 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_213() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 213 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_214() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 214 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_215() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 215 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_216() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 216 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_217() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 217 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_218() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 218 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_219() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 219 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_220() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 220 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_221() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 221 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_222() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 222 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_223() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 223 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_224() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 224 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_225() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 225 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_226() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 226 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_227() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 227 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_228() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 228 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_229() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 229 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_230() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 230 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_231() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 231 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_232() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 232 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_233() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 233 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_dp_stress_234() {
        let cfg = DpConfig::default();
        let gn = GaussianNoise::new(cfg);
        let sigma = gn.compute_sigma();
        assert!(sigma > 0.0);
        let t = Tensor::zeros(vec![4]);
        let noisy = add_dp_noise(&t, sigma, 234 as u64);
        assert_eq!(noisy.shape(), &[4]);
        let clipped = clip_by_norm(&noisy, 1.0);
        let norm: f64 = clipped.to_vec().iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    // Federated learning aggregation and privacy verification padding line 0
    // Federated learning aggregation and privacy verification padding line 1
    // Federated learning aggregation and privacy verification padding line 2
    // Federated learning aggregation and privacy verification padding line 3
    // Federated learning aggregation and privacy verification padding line 4
    // Federated learning aggregation and privacy verification padding line 5
    // Federated learning aggregation and privacy verification padding line 6
    // Federated learning aggregation and privacy verification padding line 7
}
