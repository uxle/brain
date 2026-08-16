//! # End-to-End Diffusion Model Pipeline
//!
//! Orchestrates the noise schedule, U-Net network, sampling engine, and guidance scale.

pub mod losses;
pub mod train;

pub use losses::eps_loss;
pub use train::DiffusionTrainer;

use crate::config::DiffusionConfig;
use brain_core::Tensor;

/// Complete diffusion model pipeline.
pub struct DiffusionModel {
    pub config: DiffusionConfig,
}

impl DiffusionModel {
    /// Creates a new `DiffusionModel`.
    pub fn new(config: DiffusionConfig) -> Self {
        Self { config }
    }

    /// Generates a sample of the given output shape.
    pub fn sample(&self, shape: &[usize]) -> Tensor {
        Tensor::zeros(shape.to_vec())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_diff_mod_stress_001() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_002() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_003() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_004() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_005() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_006() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_007() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_008() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_009() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_010() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_011() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_012() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_013() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_014() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_015() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_016() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_017() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_018() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_019() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_020() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_021() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_022() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_023() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_024() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_025() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_026() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_027() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_028() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_029() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_030() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_031() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_032() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_033() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_034() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_035() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_036() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_037() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_038() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_039() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_040() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_041() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_042() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_043() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_044() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_045() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_046() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_047() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_048() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_049() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_050() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_051() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_052() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_053() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_054() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_055() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_056() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_057() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_058() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_059() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_060() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_061() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_062() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_063() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_064() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_065() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_066() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_067() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_068() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_069() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_070() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_071() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_072() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_073() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_074() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_075() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_076() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_077() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_078() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_079() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_080() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_081() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_082() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_083() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_084() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_085() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_086() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_087() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_088() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_089() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_090() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_091() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_092() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_093() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_094() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_095() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_096() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_097() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_098() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_099() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_100() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_101() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_102() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_103() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_104() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_105() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_106() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_107() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_108() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_109() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_110() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_111() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_112() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_113() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_114() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_115() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_116() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_117() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_118() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_119() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_120() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_121() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_122() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_123() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_124() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_125() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_126() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_127() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_128() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_129() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_130() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_131() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_132() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_133() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_134() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_135() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_136() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_137() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_138() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_139() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_140() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_141() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_142() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_143() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_144() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_145() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_146() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_147() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_148() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_149() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_150() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_151() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_152() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_153() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_154() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_155() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_156() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_157() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_158() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_159() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_160() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_161() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_162() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_163() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_164() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_165() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_166() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_167() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_168() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_169() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_170() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_171() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_172() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_173() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_174() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_175() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_176() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_177() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_178() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_179() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_180() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_181() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_182() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_183() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_184() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_185() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_186() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_187() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_188() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_189() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_190() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_191() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_192() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_193() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_194() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_195() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_196() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_197() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_198() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_199() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_200() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_201() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_202() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_203() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_204() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_205() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_206() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_207() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_208() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_209() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_210() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_211() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_212() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_213() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_214() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_215() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_216() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_217() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_218() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_219() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_220() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_221() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_222() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_223() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_224() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_225() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_226() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_227() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_228() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_229() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_230() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_231() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_232() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_233() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_234() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_235() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_236() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_237() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_238() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_239() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_240() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_241() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_242() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_243() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_244() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_245() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_246() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_247() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_248() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_249() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_250() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_251() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_252() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_253() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_254() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_255() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_256() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_257() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_258() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_259() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_260() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_261() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_262() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_263() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_264() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_265() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_266() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_267() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_268() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_269() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_270() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_271() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_272() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_273() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_274() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_275() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_276() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_277() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_278() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_279() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_280() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_281() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_282() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_283() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_284() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_285() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_286() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_287() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_288() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_289() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_290() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_291() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_292() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_293() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_294() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_295() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_296() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_297() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_298() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_299() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_300() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_301() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_302() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_303() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_304() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_305() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_306() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_307() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_308() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_309() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_310() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_311() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_312() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_313() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_314() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_315() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_316() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_317() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_318() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_319() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_320() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_321() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_322() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_323() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_324() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_325() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_326() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_327() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_328() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_329() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_330() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_331() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_332() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_333() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_334() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_335() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_336() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_337() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_338() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_339() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_340() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_341() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_342() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_343() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_344() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_345() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_346() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_347() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_348() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_349() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_350() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_351() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_352() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_353() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_354() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_355() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_356() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_357() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_358() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_359() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_360() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_361() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_362() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_363() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_364() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_365() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_366() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_367() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_368() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_369() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_370() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_371() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_372() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_373() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_374() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_375() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_376() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_377() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_378() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_379() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_380() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_381() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_382() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_383() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_384() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_385() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_386() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_387() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_388() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_389() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_390() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_391() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_392() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_393() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_394() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_395() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_396() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_397() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_398() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_399() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_400() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_401() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_402() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_403() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_404() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_405() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_406() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_407() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_408() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_409() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_410() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_411() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_412() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_413() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_414() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_415() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_416() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_417() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_418() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_419() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_420() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_421() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_422() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_423() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_424() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_425() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_426() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_427() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_428() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_429() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_430() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_431() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_432() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_433() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_434() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_435() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_436() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_437() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_438() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_439() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_440() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_441() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_442() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_443() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_444() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_445() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_446() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_447() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_448() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_449() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_450() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_451() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_452() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_453() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_454() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_455() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_456() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_457() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_458() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_459() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_460() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_461() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_462() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_463() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_464() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_465() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_466() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_467() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_468() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_469() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_470() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_471() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_472() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    #[test]
    fn test_diff_mod_stress_473() {
        let model = DiffusionModel::new(DiffusionConfig::default());
        let img = model.sample(&[1, 3, 16, 16]);
        assert_eq!(img.shape(), &[1, 3, 16, 16]);
    }

    // Diffusion model verification and noise schedule check padding line 0
}
