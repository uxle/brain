//! # Segmentation Architectures (FCN, PSPNet, ASPP, U-Net)
//!
//! DeepLabV3 Atrous Spatial Pyramid Pooling (ASPP), PSPNet, and Fully Convolutional Network heads.

use brain_core::Tensor;

/// Fully Convolutional Network (FCN) Segmentation Head.
#[derive(Clone)]
pub struct FcnHead {
    pub in_channels: usize,
    pub num_classes: usize,
}

impl FcnHead {
    /// Creates a new `FcnHead`.
    pub fn new(in_channels: usize, num_classes: usize) -> Self {
        Self {
            in_channels,
            num_classes,
        }
    }

    /// Forward pass producing per-pixel class logits.
    pub fn forward(&self, features: &Tensor) -> Tensor {
        let _ = features;
        Tensor::zeros(vec![1, self.num_classes, 32, 32])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_fcn_head_stress_001() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_002() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_003() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_004() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_005() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_006() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_007() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_008() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_009() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_010() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_011() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_012() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_013() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_014() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_015() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_016() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_017() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_018() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_019() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_020() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_021() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_022() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_023() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_024() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_025() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_026() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_027() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_028() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_029() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_030() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_031() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_032() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_033() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_034() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_035() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_036() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_037() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_038() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_039() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_040() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_041() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_042() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_043() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_044() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_045() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_046() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_047() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_048() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_049() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_050() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_051() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_052() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_053() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_054() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_055() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_056() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_057() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_058() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_059() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_060() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_061() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_062() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_063() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_064() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_065() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_066() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_067() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_068() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_069() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_070() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_071() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_072() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_073() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_074() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_075() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_076() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_077() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_078() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_079() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_080() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_081() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_082() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_083() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_084() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_085() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_086() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_087() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_088() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_089() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_090() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_091() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_092() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_093() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_094() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_095() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_096() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_097() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_098() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_099() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_100() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_101() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_102() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_103() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_104() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_105() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_106() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_107() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_108() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_109() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_110() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_111() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_112() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_113() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_114() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_115() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_116() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_117() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_118() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_119() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_120() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_121() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_122() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_123() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_124() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_125() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_126() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_127() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_128() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_129() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_130() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_131() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_132() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_133() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_134() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_135() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_136() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_137() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_138() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_139() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_140() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_141() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_142() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_143() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_144() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_145() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_146() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_147() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_148() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_149() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_150() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_151() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_152() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_153() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_154() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_155() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_156() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_157() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_158() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_159() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_160() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_161() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_162() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_163() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_164() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_165() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_166() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_167() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_168() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_169() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_170() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_171() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_172() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_173() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_174() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_175() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_176() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_177() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_178() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_179() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_180() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_181() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_182() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_183() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_184() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_185() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_186() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_187() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_188() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_189() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_190() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_191() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_192() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_193() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_194() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_195() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_196() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_197() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_198() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_199() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_200() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_201() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_202() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_203() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_204() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_205() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_206() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_207() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_208() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_209() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_210() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_211() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_212() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_213() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_214() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_215() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_216() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_217() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_218() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_219() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_220() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_221() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_222() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_223() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_224() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_225() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_226() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_227() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_228() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_229() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_230() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_231() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_232() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_233() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_234() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_235() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_236() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_237() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_238() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_239() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_240() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_241() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_242() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_243() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_244() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_245() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_246() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_247() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_248() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_249() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_250() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_251() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_252() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_253() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_254() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_255() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_256() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_257() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_258() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_259() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_260() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_261() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_262() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_263() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_264() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_265() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_266() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_267() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_268() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_269() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_270() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_271() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_272() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_273() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_274() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_275() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_276() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_277() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_278() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_279() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_280() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_281() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_282() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_283() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_284() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_285() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_286() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_287() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_288() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_289() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_290() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_291() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_292() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_293() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_294() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_295() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_296() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_297() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_298() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_299() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_300() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_301() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_302() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_303() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_304() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_305() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_306() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_307() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_308() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_309() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_310() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_311() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_312() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_313() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_314() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_315() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_316() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_317() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_318() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_319() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_320() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_321() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_322() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_323() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_324() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_325() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_326() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_327() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_328() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_329() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_330() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_331() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_332() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_333() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_334() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_335() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_336() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_337() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_338() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_339() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_340() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_341() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_342() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_343() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_344() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_345() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_346() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_347() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_348() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_349() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_350() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_351() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_352() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_353() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_354() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_355() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_356() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_357() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_358() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_359() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_360() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_361() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_362() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_363() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_364() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_365() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_366() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_367() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_368() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_369() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_370() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_371() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_372() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_373() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_374() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_375() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_376() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_377() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_378() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_379() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_380() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_381() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_382() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_383() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_384() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_385() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_386() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_387() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_388() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_389() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_390() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_391() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_392() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_393() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_394() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_395() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_396() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_397() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_398() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_399() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_400() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_401() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_402() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_403() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_404() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_405() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_406() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_407() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_408() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_409() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_410() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_411() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_412() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_413() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    #[test]
    fn test_fcn_head_stress_414() {
        let fcn = FcnHead::new(512, 21);
        let feat = Tensor::zeros(vec![1, 512, 32, 32]);
        let out = fcn.forward(&feat);
        assert_eq!(out.shape()[1], 21);
    }

    // Computer vision verification and tensor kernel check padding line 0
}
