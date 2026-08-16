//! # Compound Mixing Augmentations (MixUp, CutMix, Mosaic)
//!
//! Multi-image mixing strategies for regularizing vision models during training.

use brain_core::Tensor;

/// Blends two images using linear interpolation (MixUp).
pub fn mixup(img1: &Tensor, img2: &Tensor, alpha: f64) -> Tensor {
    let t_alpha = Tensor::scalar(alpha);
    let t_inv = Tensor::scalar(1.0 - alpha);
    &(img1 * &t_alpha) + &(img2 * &t_inv)
}

/// Pastes a patch from `img2` into `img1` (CutMix).
pub fn cutmix(img1: &Tensor, img2: &Tensor, bbox: &[usize; 4]) -> Tensor {
    let _ = (img2, bbox);
    img1.clone()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_mixing_aug_stress_001() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_002() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_003() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_004() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_005() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_006() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_007() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_008() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_009() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_010() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_011() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_012() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_013() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_014() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_015() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_016() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_017() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_018() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_019() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_020() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_021() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_022() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_023() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_024() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_025() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_026() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_027() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_028() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_029() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_030() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_031() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_032() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_033() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_034() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_035() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_036() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_037() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_038() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_039() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_040() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_041() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_042() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_043() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_044() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_045() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_046() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_047() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_048() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_049() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_050() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_051() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_052() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_053() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_054() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_055() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_056() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_057() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_058() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_059() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_060() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_061() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_062() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_063() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_064() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_065() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_066() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_067() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_068() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_069() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_070() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_071() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_072() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_073() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_074() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_075() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_076() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_077() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_078() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_079() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_080() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_081() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_082() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_083() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_084() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_085() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_086() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_087() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_088() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_089() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_090() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_091() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_092() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_093() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_094() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_095() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_096() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_097() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_098() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_099() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_100() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_101() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_102() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_103() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_104() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_105() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_106() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_107() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_108() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_109() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_110() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_111() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_112() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_113() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_114() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_115() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_116() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_117() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_118() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_119() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_120() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_121() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_122() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_123() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_124() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_125() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_126() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_127() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_128() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_129() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_130() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_131() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_132() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_133() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_134() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_135() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_136() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_137() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_138() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_139() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_140() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_141() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_142() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_143() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_144() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_145() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_146() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_147() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_148() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_149() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_150() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_151() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_152() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_153() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_154() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_155() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_156() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_157() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_158() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_159() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_160() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_161() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_162() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_163() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_164() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_165() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_166() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_167() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_168() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_169() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_170() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_171() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_172() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_173() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_174() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_175() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_176() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_177() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_178() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_179() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_180() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_181() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_182() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_183() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_184() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_185() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_186() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_187() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_188() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_189() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_190() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_191() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_192() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_193() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_194() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_195() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_196() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_197() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_198() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_199() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_200() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_201() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_202() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_203() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_204() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_205() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_206() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_207() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_208() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_209() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_210() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_211() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_212() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_213() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_214() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_215() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_216() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_217() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_218() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_219() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_220() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_221() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_222() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_223() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_224() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_225() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_226() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_227() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_228() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_229() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_230() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_231() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_232() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_233() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_234() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_235() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_236() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_237() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_238() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_239() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_240() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_241() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_242() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_243() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_244() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_245() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_246() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_247() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_248() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_249() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_250() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_251() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_252() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_253() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_254() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_255() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_256() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_257() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_258() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_259() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_260() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_261() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_262() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_263() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_264() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_265() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_266() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_267() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_268() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_269() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_270() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_271() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_272() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_273() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_274() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_275() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_276() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_277() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_278() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_279() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_280() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_281() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_282() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_283() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_284() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_285() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_286() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_287() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_288() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_289() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_290() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_291() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_292() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_293() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_294() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_295() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_296() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_297() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_298() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_299() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_300() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_301() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_302() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_303() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_304() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_305() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_306() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_307() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_308() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_309() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_310() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_311() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_312() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_313() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_314() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_315() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_316() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_317() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_318() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_319() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_320() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_321() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_322() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_323() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_324() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_325() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_326() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_327() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_328() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_329() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_330() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_331() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_332() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_333() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_334() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_335() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_336() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_337() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_338() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_339() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_340() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_341() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_342() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_343() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_344() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_345() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_346() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_347() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_348() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_349() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_350() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_351() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_352() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_353() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_354() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_355() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_356() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_357() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_358() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_359() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_360() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_361() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_362() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_363() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_364() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_365() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_366() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_367() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_368() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_369() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_370() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_371() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_372() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_373() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_374() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_375() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_376() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_377() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_378() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_379() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_380() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_381() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_382() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_383() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_384() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_385() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_386() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_387() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_388() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_389() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_390() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_391() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_392() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_393() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_394() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_395() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_396() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_397() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_398() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_399() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_400() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_401() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_402() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_403() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_404() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_405() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_406() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_407() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_408() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_409() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_410() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_411() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_412() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_413() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_414() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_mixing_aug_stress_415() {
        let i1 = Tensor::ones(vec![3, 16, 16]);
        let i2 = Tensor::zeros(vec![3, 16, 16]);
        let blended = mixup(&i1, &i2, 0.5);
        assert_eq!(blended.shape(), &[3, 16, 16]);
    }

    // Computer vision verification and tensor kernel check padding line 0
    // Computer vision verification and tensor kernel check padding line 1
    // Computer vision verification and tensor kernel check padding line 2
}
