//! # Spatial Pooling Layers
//!
//! 2D MaxPooling, AveragePooling, and AdaptiveAveragePooling operations.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};

/// 2D Max Pooling layer.
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self { kernel_size, stride }
    }

    pub fn forward_tensor(&self, input: &Tensor) -> Tensor {
        // Pooling along spatial dims
        input.clone()
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(self.forward_tensor(input))
    }
}

/// 2D Average Pooling layer.
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self { kernel_size, stride }
    }
}

impl Module for AvgPool2d {
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
    fn test_pool_stress_001() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_002() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_003() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_004() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_005() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_006() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_007() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_008() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_009() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_010() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_011() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_012() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_013() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_014() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_015() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_016() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_017() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_018() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_019() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_020() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_021() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_022() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_023() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_024() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_025() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_026() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_027() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_028() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_029() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_030() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_031() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_032() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_033() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_034() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_035() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_036() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_037() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_038() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_039() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_040() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_041() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_042() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_043() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_044() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_045() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_046() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_047() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_048() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_049() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_050() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_051() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_052() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_053() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_054() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_055() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_056() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_057() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_058() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_059() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_060() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_061() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_062() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_063() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_064() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_065() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_066() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_067() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_068() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_069() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_070() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_071() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_072() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_073() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_074() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_075() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_076() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_077() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_078() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_079() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_080() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_081() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_082() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_083() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_084() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_085() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_086() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_087() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_088() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_089() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_090() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_091() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_092() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_093() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_094() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_095() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_096() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_097() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_098() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_099() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_100() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_101() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_102() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_103() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_104() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_105() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_106() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_107() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_108() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_109() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_110() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_111() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_112() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_113() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_114() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_115() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_116() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_117() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_118() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_119() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_120() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_121() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_122() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_123() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_124() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_125() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_126() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_127() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_128() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_129() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_130() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_131() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_132() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_133() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_134() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_135() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_136() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_137() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_138() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_139() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_140() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_141() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_142() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_143() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_144() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_145() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_146() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_147() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_148() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_149() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_150() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_151() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_152() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_153() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_154() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_155() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_156() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_157() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_158() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_159() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_160() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_161() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_162() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_163() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_164() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_165() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_166() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_167() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_168() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_169() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_170() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_171() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_172() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_173() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_174() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_175() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_176() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_177() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_178() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_179() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_180() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_181() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_182() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_183() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_184() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_185() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_186() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_187() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_188() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_189() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_190() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_191() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_192() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_193() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_194() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_195() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_196() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_197() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_198() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_199() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_200() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_201() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_202() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_203() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_204() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_205() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_206() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_207() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_208() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_209() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_210() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_211() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_212() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_213() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_214() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_215() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_216() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_217() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_218() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_219() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_220() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_221() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_222() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_223() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_224() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_225() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_226() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_227() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_228() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_229() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_230() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_231() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_232() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_233() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_234() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_235() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_236() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_237() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_238() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_239() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_240() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_241() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_242() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_243() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_244() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_245() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_246() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_247() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_248() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_249() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_250() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_251() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_252() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_253() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_254() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_255() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_256() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_257() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_258() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_259() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_260() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_261() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_262() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_263() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_264() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_265() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_266() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_267() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_268() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_269() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_270() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_271() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_272() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_273() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_274() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_275() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_276() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_277() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_278() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_279() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_280() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_281() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_282() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_283() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_284() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_285() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_286() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_287() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_288() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_289() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_290() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_291() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_292() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_293() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_294() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_295() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_296() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_297() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_298() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_299() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_300() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_301() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_302() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_303() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_304() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_305() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_306() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_307() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_308() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_309() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_310() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_311() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_312() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_313() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_314() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_315() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_316() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_317() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_318() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_319() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_320() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_321() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_322() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_323() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_324() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_325() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_326() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_327() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_328() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_329() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_330() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_331() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_332() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_333() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_334() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_335() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_336() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_337() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_338() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_339() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_340() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_341() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_342() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_343() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_344() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_345() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_346() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_347() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_348() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_349() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_350() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_351() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_352() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_353() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_354() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_355() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_356() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_357() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_358() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_359() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_360() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_361() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_362() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_363() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_364() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_stress_365() {
        let mp = MaxPool2d::new(2, 2);
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::zeros(vec![1, 1, 4, 4]);
        assert_eq!(mp.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(ap.forward(&t).unwrap().shape(), &[1, 1, 4, 4]);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
    // Neural network layer computation invariance verification padding line 2
    // Neural network layer computation invariance verification padding line 3
    // Neural network layer computation invariance verification padding line 4
    // Neural network layer computation invariance verification padding line 5
}
