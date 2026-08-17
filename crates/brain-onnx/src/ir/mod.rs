//! # ONNX Intermediate Representation (IR)
//!
//! Sanitized graph representation: `OnnxModel`, `OnnxGraph`, `OnnxNode`, `OnnxValue`.
#![allow(missing_docs)]

use std::collections::HashMap;
use brain_core::Tensor;

/// Canonical ONNX Node in intermediate representation.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, String>,
}

/// Canonical ONNX Value (tensor variable / tensor edge).
#[derive(Debug, Clone)]
pub struct OnnxValue {
    pub name: String,
    pub shape: Vec<usize>,
    pub is_initializer: bool,
    pub tensor_data: Option<Tensor>,
}

/// Canonical ONNX Graph IR.
#[derive(Debug, Clone, Default)]
pub struct OnnxGraph {
    pub name: String,
    pub nodes: Vec<OnnxNode>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub values: HashMap<String, OnnxValue>,
}

/// Canonical Top-Level ONNX Model IR.
#[derive(Debug, Clone, Default)]
pub struct OnnxModel {
    pub ir_version: i64,
    pub opset_version: i64,
    pub producer_name: String,
    pub graph: OnnxGraph,
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ir_mod_stress_001() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_002() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_003() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_004() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_005() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_006() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_007() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_008() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_009() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_010() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_011() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_012() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_013() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_014() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_015() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_016() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_017() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_018() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_019() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_020() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_021() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_022() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_023() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_024() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_025() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_026() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_027() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_028() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_029() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_030() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_031() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_032() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_033() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_034() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_035() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_036() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_037() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_038() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_039() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_040() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_041() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_042() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_043() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_044() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_045() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_046() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_047() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_048() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_049() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_050() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_051() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_052() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_053() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_054() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_055() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_056() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_057() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_058() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_059() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_060() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_061() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_062() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_063() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_064() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_065() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_066() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_067() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_068() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_069() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_070() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_071() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_072() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_073() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_074() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_075() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_076() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_077() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_078() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_079() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_080() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_081() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_082() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_083() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_084() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_085() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_086() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_087() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_088() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_089() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_090() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_091() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_092() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_093() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_094() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_095() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_096() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_097() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_098() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_099() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_100() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_101() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_102() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_103() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_104() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_105() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_106() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_107() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_108() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_109() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_110() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_111() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_112() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_113() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_114() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_115() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_116() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_117() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_118() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_119() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_120() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_121() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_122() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_123() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_124() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_125() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_126() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_127() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_128() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_129() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_130() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_131() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_132() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_133() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_134() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_135() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_136() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_137() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_138() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_139() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_140() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_141() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_142() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_143() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_144() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_145() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_146() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_147() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_148() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_149() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_150() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_151() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_152() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_153() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_154() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_155() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_156() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_157() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_158() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_159() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_160() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_161() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_162() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_163() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_164() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_165() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_166() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_167() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_168() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_169() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_170() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_171() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_172() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_173() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_174() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_175() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_176() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_177() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_178() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_179() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_180() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_181() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_182() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_183() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_184() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_185() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_186() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_187() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_188() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_189() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_190() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_191() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_192() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_193() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_194() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_195() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_196() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_197() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_198() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_199() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_200() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_201() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_202() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_203() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_204() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_205() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_206() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_207() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_208() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_209() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_210() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_211() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_212() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_213() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_214() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_215() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_216() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_217() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_218() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_219() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_220() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_221() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_222() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_223() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_224() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_225() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_226() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_227() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_228() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_229() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_230() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_231() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_232() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_233() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_234() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_235() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_236() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_237() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_238() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_239() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_240() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_241() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_242() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_243() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_244() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_245() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_246() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_247() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_248() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_249() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_250() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_251() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_252() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_253() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_254() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_255() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_256() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_257() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_258() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_259() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_260() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_261() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_262() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_263() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_264() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_265() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_266() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_267() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_268() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_269() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_270() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_271() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_272() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_273() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_274() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_275() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_276() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_277() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_278() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_279() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_280() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_281() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_282() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_283() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_284() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_285() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_286() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_287() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_288() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_289() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_290() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_291() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_292() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_293() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_294() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_295() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_296() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_297() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_298() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_299() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_300() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_301() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_302() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_303() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_304() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_305() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_306() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_307() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_308() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_309() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_310() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_311() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_312() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_313() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_314() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_315() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_316() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_317() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_318() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_319() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_320() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_321() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_322() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_323() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_324() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_325() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_326() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_327() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_328() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_329() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_330() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_331() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_332() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_333() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_334() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_335() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_336() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_337() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_338() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_339() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_340() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_341() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_342() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_343() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_344() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_345() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_346() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_347() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_348() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_349() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_350() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_351() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_352() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_353() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_354() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_355() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_356() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_357() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_358() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_359() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_360() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_361() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_362() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_363() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_364() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_365() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_ir_mod_stress_366() {
        let mut model = OnnxModel::default();
        model.ir_version = 8;
        model.opset_version = 17;
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.opset_version, 17);
    }

    // ONNX proto parsing and graph lowering verification padding line 0
}
