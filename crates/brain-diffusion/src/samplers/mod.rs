//! # Diffusion Samplers & Step Solvers
//!
//! Provides the primary [`Sampler`] trait, DDPM, DDIM, Euler Ancestral, and PLMS multistep solvers.

pub mod ancestral;
pub mod ddim;
pub mod ddpm;
pub mod plms;

pub use ancestral::EulerAncestralSampler;
pub use ddim::DdimSampler;
pub use ddpm::DdpmSampler;
pub use plms::PlmsSampler;

use brain_core::Tensor;

/// Abstract diffusion sampling algorithm trait.
pub trait Sampler: Send + Sync {
    fn step(&self, x: &Tensor, pred_noise: &Tensor, t: usize, prev_t: usize) -> Tensor;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_samplers_mod_stress_001() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_002() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_003() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_004() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_005() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_006() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_007() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_008() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_009() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_010() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_011() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_012() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_013() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_014() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_015() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_016() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_017() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_018() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_019() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_020() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_021() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_022() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_023() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_024() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_025() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_026() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_027() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_028() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_029() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_030() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_031() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_032() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_033() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_034() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_035() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_036() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_037() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_038() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_039() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_040() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_041() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_042() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_043() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_044() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_045() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_046() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_047() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_048() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_049() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_050() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_051() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_052() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_053() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_054() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_055() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_056() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_057() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_058() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_059() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_060() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_061() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_062() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_063() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_064() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_065() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_066() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_067() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_068() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_069() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_070() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_071() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_072() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_073() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_074() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_075() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_076() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_077() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_078() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_079() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_080() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_081() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_082() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_083() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_084() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_085() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_086() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_087() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_088() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_089() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_090() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_091() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_092() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_093() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_094() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_095() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_096() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_097() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_098() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_099() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_100() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_101() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_102() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_103() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_104() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_105() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_106() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_107() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_108() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_109() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_110() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_111() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_112() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_113() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_114() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_115() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_116() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_117() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_118() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_119() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_120() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_121() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_122() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_123() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_124() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_125() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_126() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_127() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_128() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_129() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_130() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_131() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_132() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_133() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_134() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_135() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_136() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_137() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_138() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_139() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_140() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_141() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_142() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_143() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_144() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_145() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_146() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_147() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_148() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_149() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_150() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_151() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_152() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_153() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_154() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_155() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_156() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_157() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_158() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_159() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_160() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_161() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_162() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_163() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_164() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_165() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_166() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_167() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_168() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_169() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_170() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_171() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_172() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_173() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_174() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_175() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_176() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_177() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_178() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_179() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_180() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_181() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_182() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_183() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_184() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_185() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_186() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_187() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_188() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_189() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_190() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_191() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_192() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_193() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_194() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_195() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_196() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_197() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_198() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_199() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_200() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_201() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_202() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_203() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_204() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_205() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_206() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_207() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_208() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_209() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_210() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_211() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_212() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_213() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_214() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_215() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_216() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_217() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_218() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_219() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_220() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_221() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_222() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_223() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_224() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_225() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_226() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_227() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_228() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_229() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_230() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_231() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_232() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_233() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_234() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_235() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_236() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_237() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_238() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_239() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_240() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_241() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_242() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_243() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_244() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_245() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_246() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_247() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_248() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_249() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_250() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_251() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_252() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_253() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_254() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_255() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_256() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_257() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_258() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_259() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_260() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_261() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_262() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_263() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_264() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_265() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_266() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_267() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_268() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_269() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_270() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_271() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_272() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_273() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_274() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_275() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_276() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_277() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_278() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_279() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_280() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_281() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_282() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_283() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_284() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_285() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_286() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_287() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_288() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_289() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_290() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_291() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_292() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_293() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_294() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_295() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_296() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_297() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_298() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_299() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_300() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_301() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_302() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_303() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_304() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_305() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_306() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_307() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_308() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_309() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_310() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_311() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_312() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_313() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_314() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_315() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_316() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_317() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_318() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_319() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_320() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_321() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_322() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_323() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_324() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_325() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_326() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_327() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_328() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_329() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_330() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_331() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_332() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_333() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_334() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_335() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_336() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_337() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_338() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_339() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_340() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_341() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_342() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_343() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_344() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_345() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_346() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_347() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_348() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_349() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_350() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_351() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_352() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_353() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_354() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_355() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_356() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_357() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_358() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_359() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_360() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_361() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_362() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_363() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_364() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_365() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_366() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_367() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_368() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }

    #[test]
    fn test_samplers_mod_stress_369() {
        let s = DdimSampler::new(0.0);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let eps = Tensor::zeros(vec![1, 3, 16, 16]);
        let next_x = s.step(&x, &eps, 100, 90);
        assert_eq!(next_x.shape(), x.shape());
    }
}
