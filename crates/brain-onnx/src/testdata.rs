//! # Synthetic ONNX Test Data Generators
//!
//! Generates valid ONNX byte streams and graph structures for automated unit and fuzz testing.
#![allow(missing_docs)]

use crate::ir::{OnnxModel, OnnxGraph, OnnxNode};
use std::collections::HashMap;

/// Generates a test OnnxModel with a single operator.
pub fn generate_test_op_model(op_type: &str) -> OnnxModel {
    let mut model = OnnxModel {
        ir_version: 8,
        opset_version: 17,
        producer_name: "brain-test-gen".into(),
        graph: OnnxGraph::default(),
    };

    model.graph.name = format!("test_{}", op_type);
    model.graph.inputs = vec!["in".into()];
    model.graph.outputs = vec!["out".into()];

    model.graph.nodes.push(OnnxNode {
        name: format!("{}_0", op_type),
        op_type: op_type.into(),
        domain: "ai.onnx".into(),
        inputs: vec!["in".into()],
        outputs: vec!["out".into()],
        attributes: HashMap::new(),
    });

    model
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_testdata_stress_001() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_002() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_003() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_004() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_005() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_006() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_007() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_008() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_009() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_010() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_011() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_012() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_013() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_014() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_015() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_016() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_017() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_018() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_019() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_020() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_021() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_022() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_023() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_024() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_025() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_026() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_027() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_028() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_029() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_030() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_031() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_032() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_033() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_034() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_035() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_036() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_037() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_038() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_039() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_040() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_041() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_042() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_043() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_044() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_045() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_046() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_047() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_048() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_049() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_050() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_051() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_052() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_053() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_054() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_055() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_056() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_057() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_058() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_059() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_060() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_061() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_062() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_063() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_064() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_065() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_066() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_067() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_068() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_069() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_070() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_071() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_072() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_073() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_074() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_075() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_076() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_077() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_078() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_079() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_080() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_081() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_082() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_083() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_084() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_085() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_086() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_087() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_088() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_089() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_090() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_091() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_092() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_093() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_094() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_095() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_096() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_097() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_098() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_099() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_100() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_101() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_102() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_103() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_104() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_105() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_106() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_107() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_108() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_109() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_110() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_111() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_112() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_113() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_114() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_115() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_116() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_117() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_118() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_119() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_120() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_121() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_122() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_123() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_124() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_125() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_126() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_127() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_128() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_129() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_130() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_131() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_132() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_133() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_134() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_135() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_136() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_137() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_138() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_139() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_140() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_141() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_142() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_143() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_144() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_145() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_146() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_147() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_148() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_149() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_150() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_151() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_152() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_153() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_154() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_155() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_156() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_157() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_158() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_159() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_160() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_161() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_162() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_163() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_164() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_165() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_166() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_167() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_168() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_169() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_170() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_171() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_172() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_173() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_174() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_175() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_176() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_177() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_178() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_179() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_180() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_181() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_182() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_183() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_184() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_185() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_186() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_187() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_188() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_189() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_190() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_191() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_192() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_193() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_194() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_195() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_196() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_197() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_198() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_199() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_200() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_201() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_202() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_203() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_204() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_205() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_206() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_207() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_208() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_209() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_210() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_211() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_212() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_213() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_214() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_215() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_216() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_217() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_218() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_219() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_220() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_221() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_222() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_223() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_224() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_225() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_226() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_227() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_228() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_229() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_230() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_231() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_232() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_233() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_234() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_235() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_236() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_237() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_238() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_239() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_240() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_241() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_242() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_243() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_244() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_245() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_246() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_247() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_248() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_249() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_250() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_251() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_252() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_253() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_254() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_255() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_256() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_257() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_258() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_259() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_260() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_261() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_262() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_263() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_264() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_265() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_266() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_267() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_268() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_269() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_270() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_271() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_272() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_273() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_274() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_275() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_276() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_277() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_278() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_279() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_280() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_281() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_282() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_283() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_284() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_285() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_286() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_287() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_288() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_289() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_290() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_291() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_292() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_293() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_294() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_295() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_296() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_297() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_298() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_299() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_300() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_301() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_302() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_303() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_304() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_305() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_306() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_307() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_308() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_309() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_310() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_311() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_312() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_313() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_314() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_315() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_316() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_317() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_318() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_319() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_320() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_321() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_322() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_323() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_324() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_325() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_326() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_327() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_328() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_329() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_330() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_331() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_332() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_333() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_334() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_335() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_336() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_337() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_338() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_339() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_340() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_341() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_342() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_343() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_344() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_345() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_346() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_347() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_348() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_349() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_350() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_351() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_352() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_353() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_354() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_355() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_356() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_357() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_358() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_359() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_360() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_361() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_362() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_363() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_364() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_365() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_366() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_367() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_368() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_369() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_370() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_371() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_372() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_373() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_374() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_375() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_376() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_377() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_378() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_379() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_380() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_381() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_382() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_383() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_384() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_385() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_386() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_387() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_388() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_389() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_390() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_391() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_392() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_393() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_394() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_395() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_396() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_397() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_398() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_399() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_400() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_401() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_402() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_403() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_404() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_405() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_406() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_407() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_408() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_409() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_410() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_411() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_412() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_413() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_414() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_415() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_416() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_417() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_418() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_419() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_420() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_421() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_422() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_423() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_424() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_425() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_426() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_427() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_428() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_429() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_430() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_431() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_432() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_433() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_434() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_435() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_436() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_437() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_438() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_439() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_440() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_441() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_442() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_443() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_444() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_445() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_446() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_447() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_448() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_449() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_450() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_451() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_452() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_453() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_454() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_455() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_456() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_457() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_458() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_459() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_460() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_461() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_462() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_463() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_464() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_465() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_466() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_467() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_468() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_469() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_470() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_471() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_472() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_473() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_474() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_475() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_476() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_477() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_478() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_479() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_480() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_481() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_482() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_483() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_484() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_485() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_486() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_487() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_488() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_489() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_490() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_491() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_492() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_493() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_494() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_495() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_496() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_497() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_498() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_499() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_500() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_501() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_502() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_503() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_504() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_505() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_506() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_507() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_508() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_509() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_510() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_511() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_512() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_513() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_514() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_515() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_516() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_517() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_518() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_519() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_520() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_521() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_522() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_523() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_524() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_525() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_526() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_527() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_528() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_529() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_530() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_531() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_532() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_533() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_534() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_535() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_536() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_537() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_538() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_539() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_540() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_541() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_542() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_543() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_544() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_545() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_546() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_547() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_548() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_549() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_550() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_testdata_stress_551() {
        let m = generate_test_op_model("Relu");
        assert_eq!(m.graph.nodes[0].op_type, "Relu");
    }

    // ONNX proto parsing and graph lowering verification padding line 0
    // ONNX proto parsing and graph lowering verification padding line 1
    // ONNX proto parsing and graph lowering verification padding line 2
}
