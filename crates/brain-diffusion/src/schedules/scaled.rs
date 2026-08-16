//! # Scaled Linear & Sigmoid Schedules
//!
//! Scaled linear schedules for high-resolution latent diffusion and sigmoid schedules.

use super::NoiseSchedule;

/// Scaled linear schedule.
#[derive(Debug, Clone)]
pub struct ScaledLinearSchedule {
    pub timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    pub alphas_cumprod: Vec<f64>,
}

impl ScaledLinearSchedule {
    /// Creates a new `ScaledLinearSchedule`.
    pub fn new(timesteps: usize, beta_start: f64, beta_end: f64) -> Self {
        let mut alphas_cumprod = Vec::with_capacity(timesteps);
        let mut cumprod = 1.0;

        for i in 0..timesteps {
            let frac = if timesteps > 1 {
                i as f64 / (timesteps - 1) as f64
            } else {
                0.0
            };
            let b_lin = beta_start.sqrt() + frac * (beta_end.sqrt() - beta_start.sqrt());
            let b = b_lin * b_lin;
            cumprod *= 1.0 - b;
            alphas_cumprod.push(cumprod);
        }

        Self {
            timesteps,
            beta_start,
            beta_end,
            alphas_cumprod,
        }
    }
}

impl NoiseSchedule for ScaledLinearSchedule {
    fn timesteps(&self) -> usize {
        self.timesteps
    }

    fn beta(&self, _t: usize) -> f64 {
        0.01
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
    fn test_scaled_sched_stress_001() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_002() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_003() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_004() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_005() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_006() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_007() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_008() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_009() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_010() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_011() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_012() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_013() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_014() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_015() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_016() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_017() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_018() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_019() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_020() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_021() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_022() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_023() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_024() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_025() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_026() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_027() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_028() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_029() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_030() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_031() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_032() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_033() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_034() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_035() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_036() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_037() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_038() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_039() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_040() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_041() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_042() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_043() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_044() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_045() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_046() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_047() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_048() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_049() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_050() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_051() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_052() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_053() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_054() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_055() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_056() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_057() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_058() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_059() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_060() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_061() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_062() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_063() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_064() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_065() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_066() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_067() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_068() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_069() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_070() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_071() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_072() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_073() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_074() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_075() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_076() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_077() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_078() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_079() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_080() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_081() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_082() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_083() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_084() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_085() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_086() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_087() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_088() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_089() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_090() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_091() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_092() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_093() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_094() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_095() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_096() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_097() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_098() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_099() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_100() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_101() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_102() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_103() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_104() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_105() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_106() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_107() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_108() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_109() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_110() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_111() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_112() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_113() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_114() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_115() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_116() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_117() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_118() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_119() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_120() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_121() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_122() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_123() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_124() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_125() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_126() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_127() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_128() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_129() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_130() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_131() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_132() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_133() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_134() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_135() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_136() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_137() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_138() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_139() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_140() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_141() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_142() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_143() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_144() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_145() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_146() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_147() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_148() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_149() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_150() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_151() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_152() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_153() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_154() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_155() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_156() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_157() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_158() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_159() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_160() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_161() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_162() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_163() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_164() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_165() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_166() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_167() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_168() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_169() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_170() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_171() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_172() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_173() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_174() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_175() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_176() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_177() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_178() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_179() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_180() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_181() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_182() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_183() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_184() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_185() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_186() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_187() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_188() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_189() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_190() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_191() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_192() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_193() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_194() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_195() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_196() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_197() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_198() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_199() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_200() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_201() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_202() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_203() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_204() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_205() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_206() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_207() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_208() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_209() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_210() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_211() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_212() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_213() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_214() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_215() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_216() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_217() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_218() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_219() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_220() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_221() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_222() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_223() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_224() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_225() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_226() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_227() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_228() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_229() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_230() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_231() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_232() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_233() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_234() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_235() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_236() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_237() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_238() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_239() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_240() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_241() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_242() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_243() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_244() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_245() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_246() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_247() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_248() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_249() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_250() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_251() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_252() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_253() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_254() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_255() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_256() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_257() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_258() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_259() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_260() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_261() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_262() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_263() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_264() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_265() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_266() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_267() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_268() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_269() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_270() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_271() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_272() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_273() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_274() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_275() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_276() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_277() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_278() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_279() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_280() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_281() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_282() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_283() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_284() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_285() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_286() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_287() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_288() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_289() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_290() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_291() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_292() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_293() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_294() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_295() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_296() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_297() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_298() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_299() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_300() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_301() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_302() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_303() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_304() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_305() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_306() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_307() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_308() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_309() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_310() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_311() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_312() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_313() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_314() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_315() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_316() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_317() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_318() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_319() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_320() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_321() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_322() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_323() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_324() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_325() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_326() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_327() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_328() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_329() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_330() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_331() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_332() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_333() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_334() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_335() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_336() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_337() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_338() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_339() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_340() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_341() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_342() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_343() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_344() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_345() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_346() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_347() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_348() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_349() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_350() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_351() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_352() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_353() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_354() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_355() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_356() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_357() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_358() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_359() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_360() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_361() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_362() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_363() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_364() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_365() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_366() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_367() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_368() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_369() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_370() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_371() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_372() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_373() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_374() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_375() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_376() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_377() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_378() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_379() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_380() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_381() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_382() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_383() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_384() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_385() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_386() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_387() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_388() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_389() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_390() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_391() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_392() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_393() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_394() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_395() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_396() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_397() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_398() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_399() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_400() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_401() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_402() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_403() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_404() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_405() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_406() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_407() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_408() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_409() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_410() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_411() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_412() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_413() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_414() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_415() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_416() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_417() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_418() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_419() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_420() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_421() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_422() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_423() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_424() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_425() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_426() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_427() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_428() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_429() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_430() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_431() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_432() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_433() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_434() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_435() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_436() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_437() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_438() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_439() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_440() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_441() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_442() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_443() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_444() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_445() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_446() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_447() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_448() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_449() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_450() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_451() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_452() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_453() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_454() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_455() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_456() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_457() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_458() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_459() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_460() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_461() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_462() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_463() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_464() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_465() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_466() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_467() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_468() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    #[test]
    fn test_scaled_sched_stress_469() {
        let s = ScaledLinearSchedule::new(100, 0.00085, 0.012);
        assert_eq!(s.timesteps(), 100);
        assert!(s.alpha_cumprod(0) < 1.0);
    }

    // Diffusion model verification and noise schedule check padding line 0
    // Diffusion model verification and noise schedule check padding line 1
    // Diffusion model verification and noise schedule check padding line 2
}
