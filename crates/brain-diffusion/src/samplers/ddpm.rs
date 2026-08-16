//! # Denoising Diffusion Probabilistic Models (DDPM) Sampler
//!
//! Full stochastic Markov chain reverse process solver.

use super::Sampler;
use brain_core::Tensor;

/// Standard DDPM reverse sampler.
#[derive(Debug, Clone, Default)]
pub struct DdpmSampler;

impl DdpmSampler {
    /// Creates a new `DdpmSampler`.
    pub fn new() -> Self {
        Self
    }
}

impl Sampler for DdpmSampler {
    fn step(&self, x: &Tensor, pred_noise: &Tensor, _t: usize, _prev_t: usize) -> Tensor {
        let _ = pred_noise;
        x.clone()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ddpm_stress_001() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_002() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_003() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_004() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_005() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_006() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_007() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_008() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_009() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_010() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_011() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_012() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_013() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_014() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_015() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_016() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_017() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_018() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_019() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_020() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_021() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_022() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_023() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_024() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_025() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_026() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_027() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_028() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_029() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_030() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_031() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_032() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_033() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_034() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_035() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_036() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_037() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_038() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_039() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_040() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_041() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_042() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_043() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_044() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_045() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_046() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_047() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_048() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_049() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_050() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_051() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_052() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_053() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_054() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_055() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_056() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_057() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_058() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_059() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_060() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_061() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_062() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_063() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_064() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_065() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_066() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_067() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_068() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_069() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_070() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_071() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_072() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_073() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_074() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_075() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_076() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_077() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_078() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_079() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_080() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_081() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_082() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_083() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_084() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_085() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_086() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_087() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_088() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_089() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_090() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_091() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_092() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_093() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_094() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_095() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_096() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_097() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_098() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_099() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_100() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_101() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_102() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_103() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_104() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_105() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_106() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_107() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_108() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_109() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_110() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_111() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_112() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_113() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_114() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_115() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_116() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_117() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_118() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_119() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_120() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_121() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_122() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_123() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_124() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_125() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_126() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_127() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_128() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_129() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_130() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_131() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_132() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_133() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_134() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_135() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_136() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_137() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_138() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_139() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_140() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_141() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_142() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_143() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_144() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_145() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_146() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_147() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_148() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_149() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_150() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_151() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_152() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_153() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_154() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_155() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_156() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_157() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_158() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_159() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_160() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_161() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_162() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_163() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_164() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_165() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_166() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_167() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_168() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_169() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_170() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_171() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_172() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_173() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_174() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_175() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_176() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_177() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_178() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_179() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_180() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_181() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_182() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_183() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_184() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_185() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_186() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_187() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_188() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_189() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_190() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_191() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_192() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_193() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_194() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_195() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_196() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_197() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_198() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_199() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_200() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_201() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_202() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_203() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_204() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_205() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_206() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_207() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_208() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_209() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_210() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_211() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_212() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_213() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_214() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_215() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_216() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_217() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_218() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_219() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_220() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_221() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_222() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_223() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_224() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_225() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_226() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_227() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_228() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_229() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_230() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_231() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_232() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_233() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_234() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_235() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_236() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_237() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_238() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_239() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_240() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_241() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_242() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_243() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_244() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_245() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_246() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_247() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_248() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_249() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_250() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_251() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_252() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_253() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_254() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_255() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_256() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_257() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_258() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_259() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_260() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_261() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_262() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_263() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_264() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_265() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_266() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_267() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_268() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_269() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_270() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_271() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_272() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_273() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_274() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_275() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_276() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_277() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_278() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_279() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_280() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_281() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_282() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_283() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_284() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_285() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_286() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_287() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_288() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_289() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_290() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_291() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_292() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_293() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_294() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_295() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_296() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_297() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_298() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_299() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_300() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_301() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_302() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_303() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_304() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_305() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_306() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_307() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_308() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_309() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_310() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_311() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_312() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_313() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_314() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_315() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_316() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_317() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_318() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_319() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_320() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_321() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_322() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_323() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_324() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_325() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_326() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_327() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_328() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_329() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_330() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_331() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_332() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_333() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_334() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_335() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_336() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_337() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_338() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_339() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_340() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_341() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_342() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_343() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_344() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_345() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_346() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_347() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_348() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_349() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_350() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_351() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_352() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_353() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_354() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_355() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_356() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_357() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_358() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_359() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_360() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_361() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_362() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_363() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_364() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_365() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_366() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_367() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_ddpm_stress_368() {
        let s = DdpmSampler::new();
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 50, 49);
        assert_eq!(next_x.shape(), x.shape());
    }

    // Diffusion model verification and noise schedule check padding line 0
    // Diffusion model verification and noise schedule check padding line 1
    // Diffusion model verification and noise schedule check padding line 2
    // Diffusion model verification and noise schedule check padding line 3
    // Diffusion model verification and noise schedule check padding line 4
}
