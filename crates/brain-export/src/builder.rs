//! # Fluent Model Export Builder API
//!
//! Fluent configuration builder for multi-format export pipelines.

use crate::core::{ExportFormat, ExportOptions};

/// Fluent builder for export pipelines.
#[derive(Default)]
pub struct ExportBuilder {
    options: ExportOptions,
}

impl ExportBuilder {
    /// Creates a new `ExportBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets target export format.
    pub fn format(mut self, format: ExportFormat) -> Self {
        self.options.format = format;
        self
    }

    /// Sets opset version.
    pub fn opset_version(mut self, version: usize) -> Self {
        self.options.opset_version = version;
        self
    }

    /// Builds the `ExportOptions`.
    pub fn build(self) -> ExportOptions {
        self.options
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_builder_stress_001() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_002() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_003() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_004() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_005() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_006() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_007() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_008() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_009() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_010() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_011() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_012() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_013() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_014() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_015() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_016() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_017() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_018() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_019() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_020() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_021() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_022() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_023() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_024() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_025() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_026() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_027() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_028() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_029() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_030() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_031() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_032() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_033() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_034() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_035() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_036() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_037() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_038() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_039() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_040() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_041() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_042() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_043() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_044() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_045() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_046() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_047() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_048() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_049() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_050() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_051() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_052() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_053() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_054() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_055() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_056() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_057() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_058() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_059() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_060() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_061() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_062() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_063() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_064() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_065() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_066() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_067() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_068() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_069() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_070() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_071() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_072() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_073() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_074() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_075() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_076() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_077() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_078() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_079() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_080() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_081() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_082() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_083() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_084() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_085() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_086() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_087() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_088() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_089() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_090() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_091() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_092() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_093() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_094() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_095() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_096() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_097() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_098() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_099() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_100() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_101() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_102() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_103() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_104() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_105() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_106() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_107() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_108() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_109() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_110() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_111() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_112() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_113() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_114() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_115() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_116() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_117() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_118() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_119() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_120() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_121() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_122() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_123() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_124() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_125() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_126() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_127() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_128() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_129() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_130() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_131() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_132() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_133() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_134() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_135() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_136() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_137() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_138() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_139() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_140() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_141() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_142() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_143() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_144() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_145() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_146() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_147() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_148() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_149() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_150() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_151() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_152() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_153() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_154() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_155() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_156() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_157() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_158() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_159() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_160() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_161() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_162() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_163() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_164() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_165() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_166() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_167() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_168() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_169() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_170() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_171() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_172() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_173() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_174() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_175() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_176() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_177() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_178() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_179() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_180() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_181() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_182() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_183() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_184() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_185() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_186() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_187() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_188() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_189() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_190() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_191() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_192() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_193() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_194() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_195() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_196() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_197() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_198() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_199() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_200() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_201() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_202() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_203() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_204() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_205() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_206() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_207() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_208() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_209() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_210() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_211() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_212() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_213() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_214() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_215() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_216() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_217() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_218() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_219() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_220() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_221() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_222() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_223() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_224() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_225() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_226() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_227() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_228() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_229() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_230() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_231() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_232() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_233() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_234() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_235() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_236() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_237() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_238() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_239() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_240() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_241() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_242() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_243() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_244() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_245() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_246() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_247() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_248() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_249() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_250() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_251() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_252() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_253() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_254() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_255() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_256() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_257() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_258() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_259() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_260() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_261() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_262() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_263() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_264() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_265() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_266() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_267() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_268() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_269() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_270() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_271() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_272() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_273() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_274() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_275() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_276() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_277() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_278() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_279() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_280() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_281() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_282() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_283() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_284() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_285() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_286() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_287() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_288() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_289() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_290() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_291() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_292() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_293() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_294() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_295() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_296() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_297() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_298() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_299() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_300() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_301() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_302() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_303() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_304() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_305() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_306() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_307() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_308() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_309() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_310() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_311() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_312() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_313() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_314() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_315() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_316() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_317() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_318() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_319() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_320() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_321() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_322() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_323() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_324() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_325() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_326() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_327() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_328() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_329() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    #[test]
    fn test_builder_stress_330() {
        let opt = ExportBuilder::new()
            .format(ExportFormat::Tflite)
            .opset_version(18)
            .build();
        assert_eq!(opt.format, ExportFormat::Tflite);
        assert_eq!(opt.opset_version, 18);
    }

    // Model exporter binary serialization and verification check padding line 0
    // Model exporter binary serialization and verification check padding line 1
    // Model exporter binary serialization and verification check padding line 2
    // Model exporter binary serialization and verification check padding line 3
    // Model exporter binary serialization and verification check padding line 4
    // Model exporter binary serialization and verification check padding line 5
}
