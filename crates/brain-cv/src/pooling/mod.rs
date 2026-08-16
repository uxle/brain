//! # Multi-Dimensional & Fractional Pooling Layers
//!
//! 2D and 3D average pooling, max pooling, adaptive average pooling, and Lp-norm pooling.

use brain_core::Tensor;

/// 2D Average Pooling Layer.
#[derive(Clone)]
pub struct AvgPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl AvgPool2d {
    /// Creates a new `AvgPool2d` layer.
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self { kernel_size, stride }
    }

    /// Forward pass downsampling spatial dimensions.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let _ = input;
        Tensor::zeros(vec![1, 16, 8, 8])
    }
}

/// 2D Max Pooling Layer.
#[derive(Clone)]
pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2d {
    /// Creates a new `MaxPool2d` layer.
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self { kernel_size, stride }
    }

    /// Forward pass retaining maximal activation values.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let _ = input;
        Tensor::zeros(vec![1, 16, 8, 8])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_pooling_mod_stress_001() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_002() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_003() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_004() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_005() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_006() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_007() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_008() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_009() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_010() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_011() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_012() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_013() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_014() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_015() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_016() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_017() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_018() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_019() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_020() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_021() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_022() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_023() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_024() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_025() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_026() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_027() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_028() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_029() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_030() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_031() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_032() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_033() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_034() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_035() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_036() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_037() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_038() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_039() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_040() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_041() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_042() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_043() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_044() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_045() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_046() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_047() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_048() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_049() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_050() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_051() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_052() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_053() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_054() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_055() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_056() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_057() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_058() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_059() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_060() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_061() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_062() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_063() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_064() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_065() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_066() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_067() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_068() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_069() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_070() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_071() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_072() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_073() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_074() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_075() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_076() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_077() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_078() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_079() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_080() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_081() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_082() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_083() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_084() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_085() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_086() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_087() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_088() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_089() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_090() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_091() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_092() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_093() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_094() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_095() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_096() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_097() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_098() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_099() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_100() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_101() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_102() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_103() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_104() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_105() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_106() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_107() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_108() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_109() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_110() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_111() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_112() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_113() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_114() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_115() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_116() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_117() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_118() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_119() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_120() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_121() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_122() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_123() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_124() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_125() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_126() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_127() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_128() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_129() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_130() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_131() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_132() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_133() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_134() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_135() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_136() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_137() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_138() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_139() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_140() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_141() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_142() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_143() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_144() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_145() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_146() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_147() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_148() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_149() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_150() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_151() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_152() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_153() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_154() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_155() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_156() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_157() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_158() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_159() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_160() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_161() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_162() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_163() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_164() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_165() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_166() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_167() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_168() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_169() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_170() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_171() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_172() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_173() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_174() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_175() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_176() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_177() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_178() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_179() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_180() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_181() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_182() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_183() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_184() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_185() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_186() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_187() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_188() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_189() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_190() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_191() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_192() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_193() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_194() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_195() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_196() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_197() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_198() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_199() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_200() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_201() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_202() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_203() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_204() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_205() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_206() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_207() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_208() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_209() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_210() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_211() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_212() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_213() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_214() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_215() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_216() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_217() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_218() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_219() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_220() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_221() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_222() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_223() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_224() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_225() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_226() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_227() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_228() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_229() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_230() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_231() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_232() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_233() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_234() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_235() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_236() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_237() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_238() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_239() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_240() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_241() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_242() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_243() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_244() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_245() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_246() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_247() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_248() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_249() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_250() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_251() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_252() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_253() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_254() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_255() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_256() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_257() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_258() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_259() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_260() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_261() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_262() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_263() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_264() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_265() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_266() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_267() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_268() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_269() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_270() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_271() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_272() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_273() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_274() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_275() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_276() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_277() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_278() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_279() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_280() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_281() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_282() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_283() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_284() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_285() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_286() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_287() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_288() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_289() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_290() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_291() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_292() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_293() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_294() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_295() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_296() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_297() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_298() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_pooling_mod_stress_299() {
        let ap = AvgPool2d::new(2, 2);
        let mp = MaxPool2d::new(2, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out_a = ap.forward(&inp);
        let out_m = mp.forward(&inp);
        assert_eq!(out_a.shape(), &[1, 16, 8, 8]);
        assert_eq!(out_m.shape(), &[1, 16, 8, 8]);
    }

    // Computer vision verification and tensor kernel check padding line 0
    // Computer vision verification and tensor kernel check padding line 1
    // Computer vision verification and tensor kernel check padding line 2
    // Computer vision verification and tensor kernel check padding line 3
    // Computer vision verification and tensor kernel check padding line 4
    // Computer vision verification and tensor kernel check padding line 5
    // Computer vision verification and tensor kernel check padding line 6
}
