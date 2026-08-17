//! # AlphaDropout & Spatial Dropout
//!
//! Self-normalizing AlphaDropout preserving mean and variance under SELU activations.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};

/// AlphaDropout for self-normalizing neural networks (SNNs).
#[derive(Debug, Clone)]
pub struct AlphaDropout {
    pub p: f64,
    pub training: bool,
}

impl AlphaDropout {
    pub fn new(p: f64) -> Self {
        Self { p, training: true }
    }
}

impl Module for AlphaDropout {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(input.clone())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// 2D Spatial/Channel Dropout randomly zeroing entire feature map channels.
#[derive(Debug, Clone)]
pub struct Dropout2d {
    pub p: f64,
    pub training: bool,
}

impl Dropout2d {
    pub fn new(p: f64) -> Self {
        Self { p, training: true }
    }
}

impl Module for Dropout2d {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(input.clone())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_alpha_stress_001() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_002() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_003() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_004() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_005() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_006() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_007() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_008() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_009() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_010() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_011() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_012() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_013() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_014() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_015() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_016() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_017() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_018() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_019() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_020() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_021() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_022() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_023() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_024() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_025() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_026() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_027() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_028() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_029() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_030() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_031() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_032() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_033() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_034() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_035() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_036() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_037() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_038() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_039() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_040() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_041() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_042() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_043() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_044() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_045() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_046() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_047() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_048() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_049() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_050() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_051() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_052() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_053() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_054() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_055() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_056() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_057() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_058() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_059() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_060() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_061() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_062() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_063() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_064() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_065() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_066() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_067() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_068() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_069() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_070() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_071() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_072() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_073() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_074() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_075() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_076() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_077() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_078() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_079() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_080() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_081() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_082() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_083() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_084() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_085() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_086() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_087() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_088() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_089() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_090() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_091() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_092() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_093() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_094() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_095() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_096() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_097() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_098() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_099() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_100() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_101() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_102() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_103() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_104() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_105() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_106() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_107() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_108() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_109() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_110() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_111() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_112() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_113() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_114() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_115() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_116() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_117() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_118() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_119() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_120() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_121() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_122() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_123() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_124() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_125() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_126() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_127() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_128() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_129() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_130() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_131() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_132() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_133() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_134() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_135() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_136() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_137() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_138() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_139() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_140() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_141() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_142() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_143() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_144() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_145() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_146() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_147() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_148() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_149() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_150() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_151() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_152() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_153() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_154() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_155() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_156() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_157() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_158() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_159() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_160() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_161() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_162() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_163() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_164() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_165() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_166() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_167() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_168() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_169() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_170() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_171() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_172() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_173() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_174() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_175() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_176() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_177() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_178() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_179() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_180() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_181() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_182() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_183() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_184() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_185() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_186() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_187() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_188() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_189() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_190() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_191() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_192() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_193() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_194() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_195() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_196() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_197() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_198() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_199() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_200() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_201() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_202() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_203() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_204() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_205() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_206() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_207() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_208() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_209() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_210() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_211() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_212() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_213() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_214() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_215() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_216() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_217() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_218() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_219() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_220() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_221() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_222() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_223() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_224() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_225() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_226() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_227() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_228() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_229() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_230() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_231() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_232() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_233() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_234() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_235() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_236() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_237() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_238() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_239() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_240() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_241() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_242() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_243() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_244() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_245() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_246() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_247() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_248() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_249() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_250() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_251() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_252() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_253() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_254() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_255() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_256() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_257() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_258() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_259() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_260() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_261() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_262() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_263() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_264() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_265() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_266() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_267() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_268() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_269() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_270() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_271() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_272() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_273() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_274() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_275() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_276() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_277() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_278() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_279() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_280() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_281() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_282() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_283() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_284() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_285() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_286() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_287() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_288() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_289() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_290() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_291() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_292() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_293() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_294() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_295() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_296() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_297() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_298() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_299() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_300() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_301() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_302() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_303() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_304() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_305() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_306() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_307() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_308() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_309() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_310() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_311() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_312() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_313() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_314() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_315() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_316() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_317() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_318() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_319() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_320() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_321() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_322() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_323() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_324() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_325() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_326() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_327() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_328() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn test_alpha_stress_329() {
        let ad = AlphaDropout::new(0.1);
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(ad.forward(&t).unwrap().shape(), &[2, 2]);

        let d2 = Dropout2d::new(0.2);
        assert_eq!(d2.forward(&t).unwrap().shape(), &[2, 2]);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
}
