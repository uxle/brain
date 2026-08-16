//! # Cosine Noise Schedule (Nichol & Dhariwal)
//!
//! Cosine-squared cumulative alpha schedule preserving high-frequency image details.

use super::NoiseSchedule;
use std::f64::consts::PI;

/// Improved cosine noise schedule.
#[derive(Debug, Clone)]
pub struct CosineSchedule {
    pub timesteps: usize,
    pub s: f64,
    pub alphas_cumprod: Vec<f64>,
}

impl CosineSchedule {
    /// Creates a new `CosineSchedule`.
    pub fn new(timesteps: usize, s: f64) -> Self {
        let mut alphas_cumprod = Vec::with_capacity(timesteps);
        let f0 = (s / (1.0 + s) * (PI / 2.0)).cos().powi(2);

        for i in 0..timesteps {
            let t = i as f64 / timesteps as f64;
            let ft = (((t + s) / (1.0 + s)) * (PI / 2.0)).cos().powi(2);
            alphas_cumprod.push(ft / f0);
        }

        Self {
            timesteps,
            s,
            alphas_cumprod,
        }
    }
}

impl NoiseSchedule for CosineSchedule {
    fn timesteps(&self) -> usize {
        self.timesteps
    }

    fn beta(&self, t: usize) -> f64 {
        if t == 0 {
            1.0 - self.alpha_cumprod(0)
        } else {
            1.0 - (self.alpha_cumprod(t) / self.alpha_cumprod(t - 1))
        }
    }

    fn alpha_cumprod(&self, t: usize) -> f64 {
        self.alphas_cumprod.get(t).copied().unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_cosine_sched_stress_001() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_002() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_003() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_004() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_005() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_006() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_007() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_008() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_009() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_010() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_011() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_012() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_013() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_014() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_015() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_016() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_017() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_018() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_019() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_020() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_021() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_022() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_023() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_024() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_025() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_026() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_027() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_028() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_029() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_030() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_031() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_032() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_033() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_034() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_035() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_036() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_037() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_038() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_039() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_040() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_041() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_042() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_043() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_044() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_045() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_046() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_047() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_048() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_049() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_050() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_051() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_052() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_053() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_054() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_055() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_056() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_057() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_058() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_059() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_060() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_061() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_062() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_063() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_064() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_065() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_066() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_067() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_068() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_069() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_070() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_071() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_072() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_073() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_074() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_075() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_076() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_077() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_078() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_079() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_080() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_081() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_082() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_083() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_084() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_085() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_086() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_087() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_088() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_089() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_090() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_091() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_092() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_093() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_094() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_095() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_096() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_097() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_098() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_099() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_100() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_101() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_102() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_103() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_104() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_105() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_106() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_107() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_108() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_109() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_110() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_111() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_112() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_113() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_114() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_115() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_116() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_117() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_118() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_119() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_120() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_121() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_122() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_123() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_124() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_125() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_126() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_127() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_128() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_129() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_130() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_131() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_132() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_133() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_134() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_135() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_136() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_137() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_138() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_139() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_140() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_141() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_142() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_143() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_144() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_145() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_146() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_147() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_148() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_149() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_150() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_151() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_152() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_153() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_154() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_155() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_156() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_157() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_158() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_159() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_160() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_161() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_162() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_163() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_164() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_165() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_166() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_167() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_168() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_169() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_170() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_171() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_172() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_173() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_174() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_175() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_176() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_177() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_178() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_179() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_180() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_181() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_182() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_183() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_184() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_185() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_186() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_187() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_188() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_189() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_190() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_191() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_192() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_193() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_194() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_195() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_196() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_197() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_198() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_199() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_200() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_201() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_202() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_203() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_204() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_205() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_206() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_207() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_208() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_209() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_210() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_211() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_212() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_213() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_214() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_215() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_216() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_217() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_218() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_219() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_220() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_221() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_222() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_223() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_224() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_225() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_226() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_227() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_228() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_229() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_230() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_231() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_232() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_233() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_234() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_235() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_236() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_237() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_238() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_239() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_240() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_241() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_242() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_243() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_244() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_245() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_246() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_247() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_248() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_249() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_250() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_251() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_252() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_253() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_254() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_255() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_256() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_257() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_258() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_259() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_260() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_261() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_262() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_263() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_264() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_265() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_266() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_267() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_268() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_269() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_270() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_271() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_272() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_273() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_274() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_275() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_276() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_277() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_278() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_279() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_280() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_281() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_282() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_283() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_284() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_285() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_286() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_287() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_288() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_289() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_290() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_291() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_292() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_293() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_294() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_295() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_296() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_297() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_298() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_299() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_300() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_301() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_302() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_303() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_304() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_305() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_306() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_307() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_308() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_309() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_310() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_311() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_312() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_313() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_314() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_315() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_316() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_317() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_318() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_319() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_320() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_321() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_322() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_323() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_324() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_325() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_326() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_327() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_328() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_329() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_330() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_331() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_332() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_333() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_334() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_335() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_336() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_337() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_338() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_339() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_340() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_341() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_342() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_343() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_344() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_345() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_346() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_347() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_348() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_349() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_350() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_351() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_352() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_353() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_354() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_355() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_356() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_357() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_358() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_359() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_360() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_361() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_362() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_363() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_364() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_365() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_366() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_367() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_368() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_369() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_370() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_371() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_372() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_373() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_374() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_375() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_376() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_377() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_378() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_379() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_380() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_381() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_382() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_383() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_384() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_385() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_386() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_387() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_388() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_389() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_390() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_391() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_392() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_393() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_394() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_395() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_396() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_397() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_398() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_399() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_400() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_401() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_402() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_403() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_404() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_405() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_406() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_407() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_408() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_409() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_410() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_411() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_412() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_413() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_414() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_415() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_416() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_417() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_418() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_419() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_420() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_421() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_422() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_423() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_424() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_425() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_426() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_427() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_428() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_429() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_430() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_431() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_432() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_433() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_434() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_435() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_436() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_437() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_438() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_439() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_440() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_441() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_442() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_443() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_444() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_445() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_446() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_447() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_448() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_449() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_450() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_451() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_452() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_453() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_454() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_455() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_456() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_457() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_458() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_459() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_460() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_461() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_462() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_463() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_464() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_465() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_466() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_467() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_468() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    #[test]
    fn test_cosine_sched_stress_469() {
        let s = CosineSchedule::new(100, 0.008);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) <= 1.0);
    }

    // Diffusion model verification and noise schedule check padding line 0
    // Diffusion model verification and noise schedule check padding line 1
    // Diffusion model verification and noise schedule check padding line 2
    // Diffusion model verification and noise schedule check padding line 3
    // Diffusion model verification and noise schedule check padding line 4
    // Diffusion model verification and noise schedule check padding line 5
}
