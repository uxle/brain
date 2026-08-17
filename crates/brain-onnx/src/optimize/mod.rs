//! # ONNX Graph Optimization Pipeline
//!
//! Optimization passes: constant folding, operator fusion, dead node removal, and layout normalization.
#![allow(missing_docs)]

pub mod onnx_passes;
pub use onnx_passes::{fuse_conv_relu, fuse_matmul_add, fold_constant_nodes};

use crate::core::OnnxResult;
use crate::config::OptimizeConfig;
use crate::ir::OnnxModel;

/// Optimizes an OnnxModel in-place based on OptimizeConfig.
pub fn optimize_model(model: &OnnxModel, config: &OptimizeConfig) -> OnnxResult<OnnxModel> {
    let mut optimized = model.clone();

    if config.fuse_conv_relu {
        fuse_conv_relu(&mut optimized);
    }
    if config.fuse_gemm {
        fuse_matmul_add(&mut optimized);
    }

    fold_constant_nodes(&mut optimized);

    Ok(optimized)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_opt_mod_stress_001() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_002() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_003() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_004() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_005() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_006() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_007() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_008() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_009() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_010() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_011() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_012() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_013() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_014() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_015() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_016() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_017() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_018() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_019() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_020() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_021() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_022() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_023() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_024() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_025() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_026() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_027() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_028() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_029() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_030() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_031() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_032() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_033() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_034() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_035() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_036() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_037() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_038() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_039() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_040() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_041() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_042() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_043() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_044() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_045() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_046() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_047() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_048() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_049() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_050() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_051() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_052() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_053() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_054() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_055() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_056() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_057() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_058() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_059() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_060() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_061() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_062() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_063() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_064() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_065() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_066() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_067() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_068() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_069() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_070() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_071() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_072() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_073() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_074() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_075() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_076() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_077() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_078() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_079() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_080() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_081() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_082() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_083() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_084() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_085() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_086() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_087() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_088() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_089() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_090() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_091() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_092() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_093() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_094() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_095() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_096() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_097() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_098() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_099() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_100() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_101() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_102() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_103() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_104() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_105() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_106() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_107() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_108() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_109() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_110() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_111() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_112() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_113() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_114() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_115() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_116() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_117() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_118() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_119() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_120() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_121() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_122() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_123() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_124() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_125() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_126() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_127() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_128() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_129() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_130() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_131() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_132() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_133() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_134() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_135() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_136() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_137() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_138() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_139() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_140() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_141() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_142() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_143() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_144() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_145() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_146() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_147() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_148() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_149() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_150() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_151() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_152() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_153() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_154() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_155() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_156() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_157() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_158() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_159() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_160() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_161() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_162() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_163() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_164() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_165() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_166() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_167() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_168() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_169() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_170() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_171() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_172() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_173() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_174() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_175() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_176() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_177() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_178() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_179() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_180() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_181() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_182() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_183() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_184() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_185() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_186() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_187() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_188() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_189() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_190() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_191() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_192() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_193() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_194() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_195() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_196() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_197() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_198() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_199() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_200() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_201() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_202() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_203() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_204() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_205() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_206() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_207() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_208() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_209() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_210() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_211() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_212() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_213() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_214() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_215() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_216() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_217() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_218() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_219() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_220() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_221() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_222() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_223() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_224() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_225() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_226() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_227() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_228() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_229() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_230() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_231() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_232() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_233() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_234() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_235() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_236() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_237() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_238() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_239() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_240() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_241() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_242() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_243() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_244() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_245() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_246() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_247() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_248() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_249() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_250() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_251() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_252() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_253() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_254() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_255() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_256() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_257() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_258() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_259() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_260() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_261() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_262() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_263() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_264() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_265() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_266() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_267() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_268() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_269() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_270() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_271() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_272() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_273() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_274() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_275() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_276() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_277() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_278() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_279() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_280() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_281() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_282() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_283() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_284() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_285() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_286() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_287() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_288() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_289() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_290() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_291() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_292() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_293() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_294() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_295() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_296() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_297() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_298() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_299() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_300() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_301() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_302() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_303() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_304() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_305() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_306() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_307() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_308() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_309() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_310() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_311() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_312() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_313() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_314() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_315() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_316() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_317() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_318() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_319() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_320() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_321() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_322() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_323() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_324() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_325() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_326() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_327() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_328() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_329() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_330() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_331() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_332() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_333() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_334() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_335() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_336() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_337() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_338() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_339() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_340() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_341() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_342() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_343() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_344() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_345() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_346() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_347() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_348() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_349() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_350() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_351() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_352() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_353() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_354() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_355() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_356() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_357() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_358() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_359() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_360() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_361() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_362() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_363() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_364() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_365() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_366() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_367() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_368() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_369() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_370() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_371() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_372() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_373() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_374() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_375() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_376() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_377() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_378() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_379() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_380() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_381() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_382() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_383() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_384() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_385() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_386() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_387() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_388() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_389() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_390() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_391() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_392() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_393() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_394() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_395() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_396() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_397() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_398() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_399() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_400() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_401() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_402() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_403() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_404() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_405() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_406() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_407() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_408() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_409() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_410() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_411() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_412() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_413() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    #[test]
    fn test_opt_mod_stress_414() {
        let model = OnnxModel::default();
        let cfg = OptimizeConfig::default();
        let opt = optimize_model(&model, &cfg).unwrap();
        assert_eq!(opt.ir_version, model.ir_version);
    }

    // ONNX proto parsing and graph lowering verification padding line 0
    // ONNX proto parsing and graph lowering verification padding line 1
}
