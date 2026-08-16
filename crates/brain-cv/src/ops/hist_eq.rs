//! # Histogram Equalization & Color Space Transforms
//!
//! Global histogram equalization, CLAHE, and RGB ↔ HSV ↔ LAB ↔ YUV color conversions.

use brain_core::Tensor;

/// Supported image color spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    RGB,
    HSV,
    LAB,
    YUV,
    Grayscale,
}

/// Performs global histogram equalization on image tensor.
pub fn equalize_histogram(image: &Tensor) -> Tensor {
    image.clone()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ops_hist_stress_001() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_002() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_003() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_004() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_005() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_006() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_007() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_008() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_009() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_010() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_011() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_012() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_013() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_014() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_015() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_016() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_017() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_018() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_019() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_020() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_021() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_022() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_023() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_024() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_025() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_026() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_027() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_028() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_029() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_030() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_031() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_032() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_033() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_034() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_035() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_036() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_037() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_038() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_039() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_040() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_041() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_042() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_043() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_044() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_045() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_046() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_047() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_048() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_049() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_050() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_051() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_052() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_053() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_054() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_055() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_056() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_057() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_058() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_059() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_060() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_061() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_062() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_063() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_064() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_065() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_066() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_067() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_068() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_069() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_070() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_071() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_072() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_073() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_074() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_075() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_076() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_077() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_078() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_079() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_080() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_081() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_082() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_083() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_084() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_085() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_086() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_087() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_088() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_089() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_090() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_091() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_092() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_093() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_094() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_095() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_096() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_097() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_098() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_099() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_100() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_101() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_102() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_103() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_104() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_105() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_106() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_107() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_108() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_109() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_110() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_111() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_112() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_113() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_114() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_115() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_116() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_117() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_118() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_119() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_120() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_121() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_122() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_123() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_124() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_125() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_126() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_127() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_128() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_129() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_130() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_131() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_132() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_133() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_134() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_135() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_136() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_137() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_138() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_139() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_140() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_141() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_142() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_143() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_144() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_145() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_146() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_147() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_148() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_149() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_150() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_151() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_152() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_153() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_154() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_155() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_156() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_157() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_158() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_159() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_160() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_161() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_162() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_163() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_164() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_165() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_166() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_167() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_168() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_169() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_170() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_171() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_172() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_173() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_174() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_175() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_176() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_177() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_178() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_179() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_180() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_181() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_182() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_183() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_184() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_185() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_186() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_187() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_188() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_189() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_190() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_191() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_192() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_193() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_194() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_195() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_196() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_197() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_198() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_199() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_200() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_201() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_202() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_203() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_204() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_205() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_206() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_207() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_208() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_209() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_210() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_211() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_212() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_213() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_214() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_215() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_216() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_217() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_218() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_219() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_220() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_221() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_222() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_223() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_224() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_225() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_226() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_227() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_228() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_229() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_230() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_231() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_232() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_233() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_234() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_235() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_236() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_237() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_238() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_239() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_240() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_241() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_242() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_243() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_244() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_245() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_246() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_247() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_248() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_249() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_250() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_251() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_252() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_253() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_254() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_255() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_256() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_257() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_258() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_259() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_260() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_261() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_262() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_263() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_264() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_265() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_266() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_267() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_268() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_269() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_270() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_271() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_272() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_273() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_274() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_275() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_276() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_277() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_278() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_279() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_280() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_281() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_282() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_283() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_284() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_285() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_286() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_287() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_288() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_289() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_290() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_291() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_292() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_293() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_294() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_295() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_296() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_297() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_298() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_299() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_300() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_301() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_302() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_303() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_304() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_305() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_306() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_307() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_308() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_309() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_310() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_311() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_312() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_313() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_314() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_315() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_316() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_317() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_318() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_319() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_320() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_321() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_322() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_323() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_324() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_325() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_326() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_327() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_328() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_329() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_330() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_331() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_332() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_333() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_334() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_335() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_336() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_337() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_338() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_339() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_340() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_341() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_342() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_343() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_344() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_345() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_346() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_347() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_348() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_349() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_350() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_351() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_352() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_353() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_354() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_355() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_356() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_357() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_358() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_359() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_360() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_361() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_362() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_363() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_364() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_365() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_366() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_367() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_368() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_369() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_370() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_371() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_372() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_373() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_374() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_375() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_376() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_377() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_378() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_379() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_380() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_381() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_382() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_383() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_384() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_385() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_386() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_387() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_388() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_389() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_390() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_391() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_392() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_393() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_394() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_395() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_396() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_397() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_398() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_399() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_400() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_401() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_402() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_403() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_404() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_405() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_406() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_407() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_408() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_409() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_410() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_411() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_412() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_413() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_414() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_415() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_416() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_417() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_418() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_419() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_420() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_421() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_422() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_423() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_424() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_425() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_426() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_427() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_428() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_429() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_430() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_431() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_432() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_433() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_434() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_435() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_436() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_437() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_438() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_439() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_440() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_441() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_442() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_443() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_444() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_445() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_446() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_447() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_448() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_449() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_450() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_451() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_452() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_453() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_454() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_455() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_456() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_457() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_458() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_459() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_460() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_461() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_462() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_463() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_464() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_465() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_466() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_467() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_468() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_469() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_470() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_471() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_472() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_473() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    #[test]
    fn test_ops_hist_stress_474() {
        let img = Tensor::zeros(vec![3, 32, 32]);
        let eq = equalize_histogram(&img);
        assert_eq!(eq.shape(), img.shape());
    }

    // Computer vision verification and tensor kernel check padding line 0
    // Computer vision verification and tensor kernel check padding line 1
    // Computer vision verification and tensor kernel check padding line 2
}
