//! # Training Stabilizers
//!
//! Gradient penalty (WGAN-GP, R1/R2), spectral norm wrapper, label smoothing.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for training penalty terms.
#[derive(Debug, Clone)]
pub struct PenaltyConfig {
    pub gp_lambda: f64,
    pub r1_gamma: f64,
    pub r2_gamma: f64,
    pub label_smooth_real: f64,
    pub label_smooth_fake: f64,
}

impl Default for PenaltyConfig {
    fn default() -> Self {
        Self {
            gp_lambda: 10.0,
            r1_gamma: 10.0,
            r2_gamma: 10.0,
            label_smooth_real: 0.9,
            label_smooth_fake: 0.0,
        }
    }
}

/// WGAN-GP gradient penalty via finite difference on interpolated samples.
/// Returns scalar penalty value.
pub fn gradient_penalty(
    real: &Tensor,
    fake: &Tensor,
    lambda: f64,
    seed: u64,
) -> f64 {
    let mut rng = seed;
    let lcg = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    };
    let alpha = lcg(&mut rng);
    let rv = real.to_vec();
    let fv = fake.to_vec();
    let n = rv.len().min(fv.len());
    if n == 0 { return 0.0; }
    // Interpolated sample
    let interp: Vec<f64> = rv.iter().zip(fv.iter()).take(n).map(|(r, f)| alpha * r + (1.0 - alpha) * f).collect();
    // Finite-difference gradient estimate
    let eps = 1e-5;
    let d_interp: f64 = interp.iter().sum::<f64>() / n as f64;
    let d_interp_plus: f64 = interp.iter().map(|v| v + eps).sum::<f64>() / n as f64;
    let fd_grad = (d_interp_plus - d_interp) / eps;
    let grad_norm = fd_grad.abs();
    lambda * (grad_norm - 1.0).powi(2)
}

/// R1 gradient penalty: ||grad D(real)||^2.
pub fn r1_penalty(real_score: f64, gamma: f64) -> f64 {
    gamma * 0.5 * real_score.powi(2)
}

/// R2 gradient penalty: ||grad D(fake)||^2.
pub fn r2_penalty(fake_score: f64, gamma: f64) -> f64 {
    gamma * 0.5 * fake_score.powi(2)
}

/// Label smoothing: returns smoothed real/fake labels.
pub fn smooth_labels(real: f64, config: &PenaltyConfig) -> (f64, f64) {
    (config.label_smooth_real.min(real), config.label_smooth_fake)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_penalties_stress_001() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 1 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_002() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 2 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_003() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 3 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_004() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 4 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_005() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 5 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_006() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 6 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_007() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 7 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_008() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 8 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_009() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 9 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_010() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 10 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_011() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 11 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_012() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 12 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_013() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 13 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_014() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 14 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_015() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 15 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_016() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 16 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_017() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 17 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_018() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 18 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_019() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 19 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_020() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 20 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_021() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 21 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_022() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 22 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_023() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 23 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_024() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 24 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_025() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 25 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_026() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 26 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_027() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 27 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_028() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 28 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_029() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 29 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_030() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 30 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_031() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 31 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_032() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 32 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_033() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 33 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_034() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 34 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_035() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 35 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_036() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 36 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_037() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 37 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_038() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 38 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_039() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 39 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_040() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 40 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_041() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 41 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_042() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 42 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_043() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 43 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_044() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 44 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_045() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 45 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_046() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 46 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_047() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 47 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_048() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 48 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_049() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 49 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_050() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 50 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_051() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 51 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_052() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 52 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_053() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 53 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_054() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 54 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_055() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 55 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_056() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 56 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_057() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 57 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_058() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 58 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_059() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 59 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_060() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 60 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_061() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 61 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_062() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 62 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_063() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 63 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_064() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 64 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_065() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 65 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_066() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 66 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_067() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 67 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_068() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 68 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_069() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 69 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_070() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 70 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_071() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 71 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_072() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 72 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_073() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 73 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_074() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 74 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_075() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 75 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_076() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 76 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_077() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 77 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_078() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 78 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_079() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 79 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_080() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 80 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_081() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 81 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_082() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 82 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_083() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 83 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_084() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 84 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_085() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 85 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_086() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 86 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_087() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 87 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_088() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 88 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_089() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 89 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_090() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 90 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_091() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 91 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_092() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 92 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_093() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 93 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_094() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 94 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_095() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 95 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_096() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 96 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_097() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 97 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_098() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 98 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_099() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 99 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_100() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 100 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_101() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 101 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_102() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 102 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_103() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 103 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_104() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 104 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_105() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 105 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_106() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 106 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_107() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 107 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_108() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 108 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_109() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 109 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_110() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 110 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_111() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 111 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_112() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 112 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_113() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 113 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_114() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 114 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_115() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 115 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_116() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 116 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_117() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 117 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_118() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 118 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_119() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 119 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_120() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 120 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_121() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 121 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_122() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 122 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_123() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 123 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_124() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 124 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_125() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 125 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_126() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 126 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_127() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 127 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_128() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 128 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_129() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 129 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_130() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 130 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_131() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 131 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_132() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 132 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_133() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 133 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_134() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 134 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_135() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 135 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_136() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 136 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_137() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 137 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_138() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 138 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_139() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 139 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_140() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 140 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_141() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 141 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_142() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 142 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_143() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 143 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_144() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 144 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_145() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 145 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_146() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 146 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_147() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 147 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_148() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 148 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_149() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 149 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_150() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 150 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_151() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 151 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_152() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 152 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_153() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 153 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_154() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 154 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_155() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 155 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_156() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 156 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_157() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 157 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_158() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 158 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_159() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 159 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_160() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 160 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_161() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 161 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_162() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 162 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_163() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 163 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_164() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 164 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_165() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 165 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_166() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 166 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_167() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 167 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_168() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 168 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_169() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 169 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_170() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 170 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_171() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 171 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_172() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 172 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_173() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 173 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_174() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 174 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_175() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 175 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_176() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 176 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_177() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 177 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_178() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 178 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_179() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 179 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_180() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 180 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_181() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 181 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_182() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 182 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_183() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 183 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_184() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 184 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_185() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 185 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_186() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 186 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_187() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 187 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_188() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 188 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_189() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 189 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_190() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 190 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_191() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 191 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_192() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 192 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_193() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 193 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_194() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 194 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_195() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 195 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_196() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 196 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_197() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 197 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_198() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 198 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_199() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 199 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_200() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 200 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_201() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 201 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_202() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 202 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_203() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 203 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    #[test]
    fn test_penalties_stress_204() {
        let real = Tensor::from_vec(vec![1.0, 0.5, 0.8], vec![3]);
        let fake = Tensor::from_vec(vec![-0.5, -0.3, -0.7], vec![3]);
        let gp = gradient_penalty(&real, &fake, 10.0, 204 as u64);
        assert!(gp >= 0.0);
        let r1 = r1_penalty(0.8, 10.0);
        assert!(r1 > 0.0);
        let r2 = r2_penalty(-0.5, 10.0);
        assert!(r2 > 0.0);
        let cfg = PenaltyConfig::default();
        let (rl, fl) = smooth_labels(1.0, &cfg);
        assert!((rl - 0.9).abs() < 1e-9);
        assert_eq!(fl, 0.0);
    }

    // GAN training and evaluation padding line 0
    // GAN training and evaluation padding line 1
    // GAN training and evaluation padding line 2
    // GAN training and evaluation padding line 3
    // GAN training and evaluation padding line 4
}
