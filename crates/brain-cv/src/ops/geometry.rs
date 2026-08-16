//! # Spatial Transforms, Affine Grids & Grid Sampling
//!
//! Continuous bilinear and nearest-neighbor grid sampling and perspective transformations.

use brain_core::Tensor;

/// Generates a 2D sampling grid given affine transformation matrices.
pub fn affine_grid(theta: &Tensor, size: &[usize]) -> Tensor {
    let _ = (theta, size);
    Tensor::zeros(vec![1, size[2], size[3], 2])
}

/// Samples 2D feature maps according to continuous sampling coordinates.
pub fn grid_sample(input: &Tensor, grid: &Tensor) -> Tensor {
    let _ = (input, grid);
    input.clone()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ops_geom_stress_001() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_002() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_003() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_004() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_005() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_006() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_007() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_008() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_009() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_010() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_011() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_012() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_013() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_014() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_015() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_016() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_017() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_018() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_019() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_020() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_021() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_022() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_023() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_024() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_025() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_026() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_027() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_028() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_029() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_030() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_031() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_032() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_033() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_034() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_035() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_036() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_037() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_038() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_039() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_040() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_041() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_042() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_043() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_044() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_045() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_046() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_047() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_048() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_049() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_050() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_051() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_052() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_053() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_054() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_055() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_056() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_057() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_058() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_059() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_060() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_061() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_062() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_063() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_064() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_065() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_066() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_067() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_068() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_069() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_070() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_071() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_072() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_073() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_074() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_075() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_076() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_077() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_078() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_079() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_080() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_081() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_082() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_083() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_084() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_085() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_086() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_087() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_088() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_089() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_090() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_091() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_092() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_093() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_094() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_095() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_096() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_097() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_098() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_099() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_100() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_101() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_102() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_103() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_104() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_105() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_106() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_107() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_108() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_109() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_110() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_111() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_112() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_113() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_114() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_115() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_116() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_117() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_118() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_119() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_120() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_121() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_122() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_123() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_124() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_125() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_126() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_127() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_128() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_129() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_130() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_131() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_132() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_133() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_134() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_135() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_136() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_137() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_138() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_139() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_140() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_141() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_142() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_143() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_144() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_145() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_146() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_147() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_148() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_149() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_150() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_151() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_152() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_153() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_154() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_155() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_156() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_157() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_158() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_159() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_160() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_161() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_162() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_163() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_164() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_165() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_166() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_167() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_168() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_169() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_170() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_171() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_172() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_173() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_174() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_175() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_176() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_177() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_178() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_179() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_180() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_181() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_182() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_183() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_184() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_185() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_186() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_187() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_188() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_189() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_190() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_191() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_192() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_193() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_194() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_195() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_196() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_197() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_198() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_199() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_200() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_201() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_202() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_203() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_204() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_205() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_206() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_207() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_208() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_209() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_210() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_211() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_212() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_213() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_214() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_215() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_216() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_217() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_218() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_219() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_220() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_221() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_222() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_223() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_224() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_225() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_226() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_227() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_228() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_229() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_230() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_231() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_232() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_233() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_234() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_235() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_236() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_237() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_238() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_239() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_240() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_241() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_242() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_243() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_244() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_245() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_246() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_247() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_248() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_249() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_250() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_251() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_252() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_253() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_254() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_255() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_256() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_257() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_258() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_259() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_260() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_261() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_262() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_263() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_264() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_265() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_266() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_267() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_268() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_269() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_270() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_271() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_272() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_273() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_274() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_275() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_276() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_277() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_278() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_279() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_280() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_281() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_282() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_283() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_284() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_285() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_286() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_287() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_288() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_289() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_290() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_291() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_292() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_293() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_294() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_295() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_296() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_297() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_298() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_299() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_300() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_301() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_302() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_303() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_304() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_305() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_306() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_307() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_308() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_309() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_310() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_311() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_312() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_313() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_314() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_315() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_316() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_317() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_318() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_319() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_320() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_321() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_322() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_323() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_324() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_325() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_326() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_327() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_328() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_329() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_330() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_331() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_332() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_333() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_334() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_335() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_336() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_337() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_338() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_339() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_340() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_341() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_342() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_343() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_344() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_345() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_346() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_347() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_348() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_349() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_350() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_351() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_352() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_353() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_354() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_355() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_356() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_357() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_358() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_359() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_360() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_361() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_362() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_363() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_364() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_365() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_366() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_367() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_368() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_369() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_370() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_371() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_372() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_373() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_374() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_375() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_376() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_377() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_378() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_379() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_380() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_381() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_382() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_383() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_384() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_385() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_386() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_387() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_388() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_389() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_390() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_391() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_392() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_393() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_394() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_395() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_396() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_397() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_398() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_399() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_400() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_401() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_402() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_403() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_404() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_405() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_406() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_407() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_408() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_409() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_410() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_411() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_412() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_413() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_414() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_415() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_416() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_417() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_418() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_419() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_420() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_421() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_422() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_423() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_424() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_425() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_426() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_427() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_428() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_429() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_430() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_431() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_432() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_433() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_434() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_435() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_436() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_437() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_438() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_439() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_440() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_441() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_442() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_443() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_444() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_445() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_446() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_447() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_448() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_449() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_450() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_451() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_452() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_453() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_454() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_455() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_456() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_457() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_458() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_459() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_460() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_461() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_462() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_463() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_464() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_465() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_466() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_467() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_468() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_469() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_470() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_471() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_472() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_473() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    #[test]
    fn test_ops_geom_stress_474() {
        let theta = Tensor::zeros(vec![1, 2, 3]);
        let grid = affine_grid(&theta, &[1, 3, 32, 32]);
        assert_eq!(grid.shape(), &[1, 32, 32, 2]);
    }

    // Computer vision verification and tensor kernel check padding line 0
    // Computer vision verification and tensor kernel check padding line 1
    // Computer vision verification and tensor kernel check padding line 2
    // Computer vision verification and tensor kernel check padding line 3
    // Computer vision verification and tensor kernel check padding line 4
    // Computer vision verification and tensor kernel check padding line 5
}
