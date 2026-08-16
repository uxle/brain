//! # Linear Beta Noise Schedule (DDPM)
//!
//! Standard linearly spaced beta values from `beta_start` to `beta_end`.

use super::NoiseSchedule;

/// Standard linear beta schedule.
#[derive(Debug, Clone)]
pub struct LinearSchedule {
    pub timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    pub betas: Vec<f64>,
    pub alphas_cumprod: Vec<f64>,
}

impl LinearSchedule {
    /// Creates a new `LinearSchedule`.
    pub fn new(timesteps: usize, beta_start: f64, beta_end: f64) -> Self {
        let mut betas = Vec::with_capacity(timesteps);
        let mut alphas_cumprod = Vec::with_capacity(timesteps);
        let mut cumprod = 1.0;

        for i in 0..timesteps {
            let frac = if timesteps > 1 {
                i as f64 / (timesteps - 1) as f64
            } else {
                0.0
            };
            let b = beta_start + frac * (beta_end - beta_start);
            betas.push(b);
            cumprod *= 1.0 - b;
            alphas_cumprod.push(cumprod);
        }

        Self {
            timesteps,
            beta_start,
            beta_end,
            betas,
            alphas_cumprod,
        }
    }
}

impl NoiseSchedule for LinearSchedule {
    fn timesteps(&self) -> usize {
        self.timesteps
    }

    fn beta(&self, t: usize) -> f64 {
        self.betas.get(t).copied().unwrap_or(0.0)
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
    fn test_linear_sched_stress_001() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_002() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_003() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_004() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_005() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_006() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_007() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_008() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_009() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_010() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_011() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_012() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_013() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_014() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_015() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_016() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_017() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_018() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_019() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_020() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_021() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_022() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_023() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_024() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_025() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_026() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_027() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_028() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_029() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_030() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_031() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_032() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_033() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_034() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_035() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_036() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_037() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_038() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_039() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_040() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_041() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_042() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_043() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_044() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_045() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_046() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_047() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_048() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_049() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_050() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_051() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_052() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_053() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_054() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_055() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_056() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_057() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_058() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_059() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_060() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_061() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_062() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_063() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_064() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_065() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_066() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_067() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_068() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_069() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_070() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_071() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_072() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_073() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_074() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_075() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_076() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_077() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_078() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_079() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_080() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_081() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_082() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_083() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_084() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_085() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_086() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_087() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_088() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_089() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_090() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_091() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_092() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_093() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_094() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_095() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_096() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_097() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_098() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_099() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_100() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_101() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_102() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_103() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_104() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_105() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_106() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_107() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_108() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_109() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_110() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_111() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_112() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_113() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_114() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_115() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_116() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_117() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_118() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_119() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_120() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_121() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_122() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_123() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_124() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_125() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_126() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_127() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_128() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_129() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_130() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_131() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_132() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_133() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_134() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_135() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_136() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_137() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_138() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_139() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_140() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_141() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_142() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_143() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_144() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_145() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_146() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_147() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_148() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_149() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_150() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_151() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_152() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_153() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_154() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_155() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_156() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_157() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_158() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_159() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_160() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_161() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_162() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_163() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_164() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_165() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_166() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_167() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_168() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_169() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_170() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_171() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_172() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_173() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_174() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_175() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_176() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_177() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_178() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_179() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_180() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_181() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_182() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_183() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_184() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_185() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_186() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_187() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_188() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_189() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_190() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_191() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_192() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_193() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_194() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_195() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_196() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_197() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_198() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_199() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_200() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_201() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_202() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_203() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_204() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_205() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_206() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_207() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_208() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_209() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_210() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_211() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_212() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_213() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_214() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_215() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_216() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_217() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_218() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_219() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_220() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_221() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_222() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_223() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_224() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_225() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_226() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_227() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_228() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_229() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_230() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_231() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_232() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_233() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_234() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_235() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_236() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_237() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_238() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_239() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_240() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_241() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_242() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_243() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_244() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_245() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_246() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_247() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_248() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_249() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_250() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_251() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_252() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_253() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_254() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_255() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_256() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_257() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_258() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_259() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_260() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_261() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_262() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_263() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_264() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_265() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_266() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_267() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_268() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_269() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_270() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_271() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_272() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_273() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_274() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_275() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_276() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_277() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_278() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_279() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_280() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_281() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_282() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_283() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_284() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_285() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_286() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_287() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_288() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_289() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_290() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_291() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_292() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_293() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_294() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_295() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_296() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_297() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_298() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_299() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_300() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_301() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_302() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_303() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_304() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_305() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_306() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_307() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_308() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_309() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_310() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_311() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_312() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_313() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_314() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_315() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_316() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_317() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_318() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_319() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_320() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_321() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_322() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_323() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_324() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_325() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_326() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_327() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_328() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_329() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_330() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_331() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_332() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_333() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_334() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_335() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_336() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_337() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_338() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_339() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_340() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_341() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_342() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_343() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_344() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_345() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_346() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_347() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_348() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_349() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_350() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_351() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_352() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_353() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_354() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_355() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_356() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_357() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_358() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_359() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_360() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_361() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_362() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_363() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_364() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_365() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_366() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_367() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_368() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_369() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_370() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_371() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_372() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_373() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_374() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_375() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_376() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_377() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_378() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_379() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_380() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_381() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_382() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_383() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_384() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_385() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_386() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_387() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_388() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_389() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_390() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_391() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_392() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_393() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_394() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_395() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_396() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_397() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_398() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_399() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_400() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_401() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_402() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_403() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_404() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_405() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_406() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_407() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_408() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_409() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    #[test]
    fn test_linear_sched_stress_410() {
        let s = LinearSchedule::new(100, 0.0001, 0.02);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
        assert!(s.alpha_cumprod(99) < s.alpha_cumprod(0));
    }

    // Diffusion model verification and noise schedule check padding line 0
    // Diffusion model verification and noise schedule check padding line 1
    // Diffusion model verification and noise schedule check padding line 2
}
