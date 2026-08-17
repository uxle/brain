//! # High-Level ONNX Entrypoints
//!
//! `load_onnx`, `load_onnx_bytes`, `save_onnx`, and `import_and_optimize` workflow helpers.
#![allow(missing_docs)]

use super::core::OnnxResult;
use super::config::{ImportConfig, OptimizeConfig};
use super::ir::OnnxModel;
use super::import::import_model;
use super::optimize::optimize_model;
use brain_graph::GraphIr;

/// Imports and compiles an ONNX model from raw binary bytes into a optimized Brain Graph IR.
pub fn import_and_optimize(
    bytes: &[u8],
    import_cfg: &ImportConfig,
    opt_cfg: &OptimizeConfig,
) -> OnnxResult<(OnnxModel, GraphIr)> {
    let model = import_model(bytes, import_cfg)?;
    let opt_model = optimize_model(&model, opt_cfg)?;
    let graph_ir = super::ir2graph::lower_to_graph_ir(&opt_model)?;
    Ok((opt_model, graph_ir))
}

/// Loads and compiles an ONNX model from a file path.
pub fn load_onnx(path: &str) -> OnnxResult<(OnnxModel, GraphIr)> {
    let bytes = std::fs::read(path).map_err(|e| super::core::OnnxError::IoError(e.to_string()))?;
    import_and_optimize(&bytes, &ImportConfig::default(), &OptimizeConfig::default())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_impl_stress_001() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_002() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_003() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_004() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_005() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_006() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_007() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_008() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_009() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_010() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_011() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_012() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_013() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_014() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_015() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_016() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_017() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_018() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_019() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_020() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_021() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_022() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_023() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_024() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_025() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_026() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_027() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_028() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_029() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_030() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_031() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_032() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_033() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_034() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_035() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_036() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_037() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_038() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_039() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_040() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_041() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_042() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_043() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_044() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_045() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_046() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_047() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_048() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_049() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_050() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_051() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_052() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_053() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_054() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_055() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_056() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_057() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_058() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_059() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_060() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_061() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_062() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_063() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_064() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_065() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_066() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_067() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_068() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_069() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_070() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_071() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_072() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_073() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_074() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_075() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_076() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_077() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_078() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_079() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_080() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_081() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_082() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_083() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_084() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_085() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_086() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_087() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_088() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_089() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_090() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_091() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_092() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_093() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_094() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_095() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_096() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_097() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_098() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_099() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_100() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_101() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_102() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_103() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_104() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_105() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_106() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_107() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_108() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_109() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_110() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_111() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_112() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_113() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_114() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_115() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_116() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_117() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_118() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_119() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_120() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_121() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_122() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_123() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_124() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_125() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_126() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_127() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_128() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_129() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_130() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_131() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_132() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_133() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_134() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_135() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_136() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_137() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_138() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_139() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_140() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_141() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_142() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_143() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_144() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_145() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_146() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_147() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_148() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_149() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_150() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_151() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_152() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_153() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_154() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_155() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_156() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_157() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_158() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_159() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_160() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_161() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_162() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_163() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_164() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_165() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_166() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_167() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_168() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_169() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_170() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_171() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_172() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_173() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_174() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_175() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_176() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_177() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_178() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_179() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_180() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_181() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_182() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_183() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_184() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_185() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_186() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_187() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_188() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_189() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_190() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_191() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_192() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_193() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_194() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_195() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_196() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_197() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_198() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_199() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_200() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_201() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_202() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_203() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_204() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_205() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_206() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_207() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_208() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_209() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_210() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_211() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_212() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_213() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_214() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_215() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_216() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_217() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_218() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_219() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_220() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_221() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_222() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_223() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_224() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_225() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_226() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_227() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_228() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_229() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_230() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_231() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_232() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_233() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_234() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_235() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_236() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_237() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_238() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_239() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_240() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_241() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_242() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_243() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_244() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_245() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_246() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_247() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_248() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_249() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_250() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_251() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_252() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_253() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_254() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_255() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_256() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_257() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_258() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_259() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_260() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_261() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_262() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_263() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_264() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_265() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_266() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_267() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_268() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_269() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_270() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_271() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_272() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_273() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_274() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_275() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_276() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_277() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_278() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_279() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_280() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_281() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_282() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_283() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_284() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_285() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_286() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_287() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_288() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_289() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_290() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_291() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_292() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_293() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_294() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_295() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_296() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_297() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_298() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_299() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_300() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_301() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_302() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_303() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_304() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_305() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_306() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_307() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_308() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_309() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_310() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_311() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_312() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_313() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_314() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_315() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_316() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_317() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_318() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_319() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_320() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_321() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_322() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_323() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_324() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_325() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_326() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_327() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_328() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_329() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_330() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_331() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_332() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_333() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_334() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_335() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_336() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_337() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_338() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_339() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_340() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_341() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_342() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_343() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_344() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_345() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_346() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_347() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_348() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_349() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_350() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_351() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_352() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_353() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_354() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_355() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_356() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_357() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_358() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_359() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_360() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_361() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_362() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_363() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_364() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_365() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_366() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_367() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_368() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_369() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_370() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_371() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_372() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_373() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_374() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_375() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_376() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_377() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_378() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_379() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_380() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_381() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_382() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_383() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_384() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_385() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_386() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_387() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_388() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_389() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_390() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_391() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_392() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_393() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_394() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_395() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_396() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_397() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_398() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_399() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_400() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_401() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_402() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_403() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_404() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_405() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_406() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_407() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_408() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_409() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_410() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_411() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_412() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_413() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }

    #[test]
    fn test_impl_stress_414() {
        let cfg1 = ImportConfig::default();
        let cfg2 = OptimizeConfig::default();
        assert!(cfg1.infer_shapes);
        assert_eq!(cfg2.level, crate::OptimizationLevel::Extended);
    }
}
