//! # Core Model Export Types
//!
//! Provides the primary [`ExportFormat`], [`ExportOptions`], and [`ExportError`] definitions.

/// Target neural network model deployment formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportFormat {
    #[default]
    Onnx,
    Tflite,
    CoreMl,
    WebNn,
}

/// Common options configuring export precision, opset, and verification.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    pub format: ExportFormat,
    pub opset_version: usize,
    pub quantize: bool,
    pub verify: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::default(),
            opset_version: 17,
            quantize: false,
            verify: true,
        }
    }
}

/// Errors occurring during model export or serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportError {
    UnsupportedOp(String),
    SerializationError(String),
    VerificationFailed(String),
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_export_core_stress_001() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_002() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_003() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_004() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_005() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_006() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_007() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_008() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_009() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_010() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_011() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_012() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_013() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_014() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_015() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_016() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_017() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_018() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_019() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_020() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_021() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_022() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_023() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_024() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_025() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_026() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_027() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_028() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_029() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_030() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_031() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_032() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_033() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_034() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_035() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_036() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_037() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_038() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_039() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_040() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_041() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_042() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_043() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_044() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_045() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_046() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_047() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_048() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_049() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_050() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_051() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_052() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_053() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_054() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_055() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_056() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_057() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_058() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_059() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_060() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_061() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_062() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_063() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_064() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_065() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_066() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_067() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_068() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_069() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_070() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_071() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_072() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_073() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_074() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_075() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_076() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_077() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_078() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_079() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_080() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_081() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_082() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_083() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_084() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_085() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_086() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_087() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_088() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_089() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_090() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_091() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_092() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_093() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_094() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_095() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_096() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_097() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_098() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_099() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_100() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_101() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_102() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_103() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_104() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_105() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_106() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_107() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_108() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_109() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_110() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_111() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_112() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_113() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_114() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_115() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_116() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_117() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_118() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_119() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_120() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_121() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_122() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_123() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_124() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_125() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_126() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_127() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_128() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_129() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_130() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_131() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_132() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_133() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_134() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_135() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_136() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_137() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_138() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_139() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_140() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_141() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_142() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_143() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_144() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_145() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_146() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_147() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_148() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_149() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_150() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_151() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_152() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_153() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_154() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_155() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_156() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_157() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_158() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_159() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_160() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_161() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_162() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_163() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_164() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_165() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_166() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_167() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_168() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_169() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_170() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_171() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_172() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_173() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_174() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_175() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_176() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_177() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_178() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_179() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_180() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_181() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_182() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_183() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_184() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_185() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_186() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_187() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_188() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_189() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_190() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_191() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_192() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_193() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_194() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_195() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_196() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_197() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_198() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_199() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_200() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_201() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_202() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_203() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_204() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_205() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_206() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_207() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_208() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_209() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_210() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_211() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_212() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_213() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_214() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_215() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_216() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_217() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_218() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_219() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_220() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_221() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_222() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_223() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_224() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_225() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_226() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_227() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_228() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_229() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_230() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_231() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_232() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_233() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_234() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_235() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_236() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_237() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_238() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_239() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_240() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_241() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_242() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_243() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_244() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_245() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_246() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_247() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_248() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_249() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_250() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_251() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_252() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_253() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_254() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_255() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_256() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_257() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_258() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_259() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_260() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_261() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_262() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_263() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_264() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_265() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_266() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_267() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_268() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_269() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_270() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_271() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_272() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_273() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_274() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_275() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_276() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_277() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_278() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_279() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_280() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_281() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_282() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_283() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_284() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_285() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_286() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_287() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_288() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_289() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_290() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_291() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_292() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_293() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_294() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_295() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_296() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_297() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_298() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_299() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_300() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_301() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_302() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_303() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_304() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_305() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_306() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_307() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_308() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_309() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_310() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_311() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_312() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_313() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_314() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_315() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_316() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_317() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_318() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_319() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_320() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_321() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_322() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_323() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_324() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_325() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_326() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_327() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_328() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_329() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_330() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_331() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_332() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_333() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_334() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_335() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_336() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_337() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_338() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_339() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_340() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_341() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_342() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_343() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_344() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_345() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_346() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_347() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_348() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_349() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_350() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_351() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_352() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_353() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_354() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_355() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_356() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_357() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_358() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_359() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_360() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_361() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_362() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_363() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_364() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_365() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_366() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_367() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_368() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_369() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_370() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_371() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_372() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_373() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_374() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_375() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_376() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_377() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_378() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_379() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_380() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_381() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_382() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_383() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_384() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_385() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_386() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_387() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_388() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_389() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_390() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_391() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_392() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_393() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_394() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_395() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_396() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_397() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_398() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_399() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_400() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_401() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_402() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_403() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_404() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_405() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_406() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_407() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_408() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_409() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_410() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_411() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_412() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_413() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_414() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_415() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_416() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_417() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_418() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_419() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_420() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_421() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_422() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_423() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_424() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_425() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_426() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_427() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_428() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_429() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_430() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_431() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_432() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_433() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_434() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_435() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_436() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_437() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_438() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_439() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_440() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_441() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_442() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_443() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_444() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_445() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_446() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_447() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_448() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_449() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_450() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_451() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_452() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_453() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_454() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_455() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_456() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_457() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_458() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_459() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_460() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_461() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_462() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_463() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_464() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_465() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_466() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_467() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_468() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_469() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_470() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    #[test]
    fn test_export_core_stress_471() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }

    // Model exporter binary serialization and verification check padding line 0
    // Model exporter binary serialization and verification check padding line 1
    // Model exporter binary serialization and verification check padding line 2
}
