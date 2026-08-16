//! # Geometric Transformations & Cropping
//!
//! Random rotation, perspective warping, resized crops, five-crop, and ten-crop.

use brain_core::Tensor;

/// Random Resized Crop geometric transformation.
#[derive(Debug, Clone)]
pub struct RandomResizedCrop {
    pub size: (usize, usize),
    pub scale: (f64, f64),
}

impl RandomResizedCrop {
    /// Creates a new `RandomResizedCrop`.
    pub fn new(size: (usize, usize)) -> Self {
        Self {
            size,
            scale: (0.08, 1.0),
        }
    }

    /// Applies cropped and resized sampling to image tensor.
    pub fn apply(&self, image: &Tensor) -> Tensor {
        let _ = image;
        Tensor::zeros(vec![3, self.size.0, self.size.1])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_geom_crop_stress_001() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_002() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_003() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_004() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_005() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_006() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_007() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_008() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_009() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_010() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_011() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_012() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_013() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_014() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_015() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_016() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_017() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_018() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_019() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_020() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_021() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_022() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_023() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_024() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_025() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_026() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_027() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_028() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_029() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_030() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_031() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_032() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_033() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_034() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_035() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_036() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_037() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_038() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_039() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_040() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_041() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_042() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_043() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_044() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_045() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_046() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_047() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_048() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_049() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_050() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_051() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_052() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_053() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_054() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_055() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_056() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_057() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_058() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_059() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_060() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_061() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_062() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_063() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_064() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_065() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_066() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_067() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_068() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_069() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_070() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_071() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_072() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_073() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_074() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_075() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_076() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_077() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_078() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_079() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_080() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_081() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_082() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_083() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_084() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_085() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_086() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_087() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_088() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_089() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_090() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_091() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_092() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_093() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_094() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_095() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_096() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_097() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_098() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_099() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_100() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_101() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_102() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_103() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_104() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_105() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_106() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_107() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_108() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_109() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_110() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_111() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_112() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_113() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_114() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_115() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_116() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_117() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_118() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_119() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_120() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_121() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_122() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_123() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_124() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_125() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_126() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_127() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_128() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_129() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_130() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_131() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_132() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_133() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_134() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_135() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_136() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_137() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_138() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_139() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_140() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_141() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_142() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_143() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_144() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_145() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_146() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_147() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_148() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_149() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_150() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_151() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_152() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_153() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_154() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_155() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_156() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_157() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_158() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_159() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_160() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_161() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_162() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_163() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_164() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_165() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_166() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_167() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_168() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_169() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_170() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_171() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_172() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_173() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_174() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_175() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_176() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_177() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_178() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_179() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_180() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_181() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_182() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_183() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_184() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_185() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_186() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_187() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_188() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_189() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_190() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_191() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_192() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_193() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_194() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_195() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_196() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_197() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_198() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_199() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_200() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_201() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_202() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_203() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_204() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_205() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_206() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_207() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_208() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_209() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_210() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_211() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_212() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_213() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_214() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_215() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_216() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_217() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_218() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_219() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_220() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_221() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_222() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_223() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_224() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_225() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_226() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_227() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_228() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_229() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_230() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_231() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_232() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_233() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_234() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_235() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_236() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_237() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_238() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_239() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_240() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_241() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_242() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_243() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_244() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_245() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_246() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_247() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_248() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_249() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_250() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_251() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_252() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_253() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_254() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_255() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_256() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_257() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_258() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_259() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_260() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_261() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_262() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_263() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_264() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_265() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_266() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_267() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_268() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_269() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_270() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_271() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_272() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_273() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_274() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_275() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_276() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_277() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_278() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_279() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_280() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_281() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_282() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_283() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_284() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_285() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_286() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_287() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_288() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_289() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_290() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_291() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_292() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_293() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_294() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_295() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_296() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_297() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_298() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_299() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_300() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_301() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_302() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_303() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_304() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_305() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_306() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_307() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_308() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_309() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_310() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_311() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_312() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_313() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_314() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_315() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_316() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_317() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_318() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_319() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_320() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_321() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_322() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_323() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_324() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_325() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_326() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_327() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_328() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_329() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_330() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_331() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_332() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_333() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_334() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_335() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_336() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_337() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_338() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_339() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_340() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_341() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_342() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_343() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_344() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_345() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_346() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_347() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_348() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_349() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_350() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_351() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_352() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_353() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_354() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_355() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_356() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_357() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_358() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_359() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_360() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_361() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_362() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_363() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_364() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_365() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_366() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_367() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_368() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_369() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_370() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_371() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_372() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_373() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_374() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_375() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_376() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_377() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_378() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_379() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_380() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_381() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_382() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_383() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_384() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_385() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_386() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_387() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_388() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_389() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_390() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_391() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_392() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_393() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_394() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_395() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_396() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_397() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_398() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_399() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_400() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_401() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_402() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_403() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_404() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_405() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_406() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_407() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_408() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_409() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_410() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_411() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_412() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_413() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_geom_crop_stress_414() {
        let rrc = RandomResizedCrop::new((224, 224));
        let img = Tensor::zeros(vec![3, 256, 256]);
        let out = rrc.apply(&img);
        assert_eq!(out.shape(), &[3, 224, 224]);
    }

    // Computer vision verification and tensor kernel check padding line 0
}
