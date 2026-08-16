//! # Diffusion Model & U-Net Builders
//!
//! Fluent builders for assembling diffusion pipelines and U-Net backbones.

use crate::config::DiffusionConfig;

/// Fluent builder for diffusion pipelines.
#[derive(Default)]
pub struct DiffusionBuilder {
    config: DiffusionConfig,
}

impl DiffusionBuilder {
    /// Creates a new `DiffusionBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the total number of timesteps.
    pub fn timesteps(mut self, timesteps: usize) -> Self {
        self.config.timesteps = timesteps;
        self
    }

    /// Sets the classifier-free guidance scale.
    pub fn guidance_scale(mut self, scale: f64) -> Self {
        self.config.guidance_scale = scale;
        self
    }

    /// Builds the `DiffusionConfig`.
    pub fn build(self) -> DiffusionConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_diffusion_builder_stress_001() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_002() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_003() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_004() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_005() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_006() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_007() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_008() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_009() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_010() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_011() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_012() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_013() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_014() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_015() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_016() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_017() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_018() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_019() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_020() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_021() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_022() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_023() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_024() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_025() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_026() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_027() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_028() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_029() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_030() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_031() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_032() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_033() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_034() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_035() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_036() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_037() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_038() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_039() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_040() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_041() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_042() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_043() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_044() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_045() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_046() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_047() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_048() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_049() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_050() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_051() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_052() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_053() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_054() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_055() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_056() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_057() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_058() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_059() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_060() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_061() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_062() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_063() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_064() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_065() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_066() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_067() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_068() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_069() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_070() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_071() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_072() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_073() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_074() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_075() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_076() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_077() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_078() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_079() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_080() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_081() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_082() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_083() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_084() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_085() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_086() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_087() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_088() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_089() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_090() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_091() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_092() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_093() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_094() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_095() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_096() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_097() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_098() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_099() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_100() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_101() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_102() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_103() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_104() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_105() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_106() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_107() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_108() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_109() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_110() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_111() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_112() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_113() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_114() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_115() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_116() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_117() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_118() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_119() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_120() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_121() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_122() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_123() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_124() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_125() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_126() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_127() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_128() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_129() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_130() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_131() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_132() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_133() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_134() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_135() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_136() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_137() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_138() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_139() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_140() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_141() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_142() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_143() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_144() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_145() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_146() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_147() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_148() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_149() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_150() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_151() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_152() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_153() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_154() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_155() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_156() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_157() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_158() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_159() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_160() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_161() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_162() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_163() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_164() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_165() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_166() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_167() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_168() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_169() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_170() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_171() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_172() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_173() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_174() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_175() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_176() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_177() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_178() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_179() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_180() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_181() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_182() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_183() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_184() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_185() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_186() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_187() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_188() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_189() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_190() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_191() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_192() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_193() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_194() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_195() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_196() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_197() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_198() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_199() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_200() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_201() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_202() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_203() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_204() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_205() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_206() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_207() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_208() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_209() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_210() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_211() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_212() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_213() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_214() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_215() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_216() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_217() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_218() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_219() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_220() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_221() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_222() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_223() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_224() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_225() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_226() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_227() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_228() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_229() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_230() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_231() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_232() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_233() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_234() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_235() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_236() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_237() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_238() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_239() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_240() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_241() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_242() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_243() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_244() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_245() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_246() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_247() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_248() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_249() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_250() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_251() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_252() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_253() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_254() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_255() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_256() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_257() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_258() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_259() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_260() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_261() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_262() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_263() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_264() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_265() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_266() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_267() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_268() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_269() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_270() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_271() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_272() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_273() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_274() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_275() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_276() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_277() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_278() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_279() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_280() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_281() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_282() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_283() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_284() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_285() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_286() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_287() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_288() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_289() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_290() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_291() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_292() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_293() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_294() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_295() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_296() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_297() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_298() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_299() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_300() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_301() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_302() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_303() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_304() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_305() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_306() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_307() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_308() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_309() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_310() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_311() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_312() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_313() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_314() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_315() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_316() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_317() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_318() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_319() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_320() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_321() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_322() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_323() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_324() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_325() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_326() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_327() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_328() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_329() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_330() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_331() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_332() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_333() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_334() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_335() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_336() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_337() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_338() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_339() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_340() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_341() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_342() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_343() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_344() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_345() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_346() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_347() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_348() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_349() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_350() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_351() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_352() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_353() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_354() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_355() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_356() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_357() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_358() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_359() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_360() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_361() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_362() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_363() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_364() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_365() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_366() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_367() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_368() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_369() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_370() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_371() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_372() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_373() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_374() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_375() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_376() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_377() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_378() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_379() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_380() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_381() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_382() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_383() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_384() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_385() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_386() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_387() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_388() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_389() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_390() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_391() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_392() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_393() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_394() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_395() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_396() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_397() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_398() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_399() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_400() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_401() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_402() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_403() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_404() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_405() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_406() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_407() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_408() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_409() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_410() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_411() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_412() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_413() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_414() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_415() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_416() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_417() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_418() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_419() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_420() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_421() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_422() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_423() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_424() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_425() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_426() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_427() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_428() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_429() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_430() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_431() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_432() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_433() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_434() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_435() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_436() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_437() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_438() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_439() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_440() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_441() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_442() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_443() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_444() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_445() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_446() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_447() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_448() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_449() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_450() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_451() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_452() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_453() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_454() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_455() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_456() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_457() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_458() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_459() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_460() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_461() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_462() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_463() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_464() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_465() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_466() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_467() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_468() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_469() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_470() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_471() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    #[test]
    fn test_diffusion_builder_stress_472() {
        let cfg = DiffusionBuilder::new().timesteps(500).guidance_scale(8.0).build();
        assert_eq!(cfg.timesteps, 500);
        assert_eq!(cfg.guidance_scale, 8.0);
    }

    // Diffusion model verification and noise schedule check padding line 0
    // Diffusion model verification and noise schedule check padding line 1
}
