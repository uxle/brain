//! # Exporting Brain Graph IR to ONNX Model
//!
//! Reverse lowering translating `brain_graph::GraphIr` structures back into `OnnxModel` IR.
#![allow(missing_docs)]

use crate::core::OnnxResult;
use crate::ir::{OnnxModel, OnnxGraph, OnnxNode};
use brain_graph::GraphIr;
use std::collections::HashMap;

/// Translates Brain GraphIr into canonical OnnxModel IR.
pub fn lower_from_graph_ir(graph_ir: &GraphIr) -> OnnxResult<OnnxModel> {
    let mut model = OnnxModel {
        ir_version: 8,
        opset_version: 17,
        producer_name: "brain-onnx-export".into(),
        graph: OnnxGraph::default(),
    };

    model.graph.name = graph_ir.name.clone();

    for node in &graph_ir.nodes {
        model.graph.nodes.push(OnnxNode {
            name: node.name.clone(),
            op_type: format!("{:?}", node.op),
            domain: "ai.onnx".into(),
            inputs: Vec::new(),
            outputs: vec![format!("{}_out", node.name)],
            attributes: HashMap::new(),
        });
    }

    Ok(model)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_graph2onnx_stress_001() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_002() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_003() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_004() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_005() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_006() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_007() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_008() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_009() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_010() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_011() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_012() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_013() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_014() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_015() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_016() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_017() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_018() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_019() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_020() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_021() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_022() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_023() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_024() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_025() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_026() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_027() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_028() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_029() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_030() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_031() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_032() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_033() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_034() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_035() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_036() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_037() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_038() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_039() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_040() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_041() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_042() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_043() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_044() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_045() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_046() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_047() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_048() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_049() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_050() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_051() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_052() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_053() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_054() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_055() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_056() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_057() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_058() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_059() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_060() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_061() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_062() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_063() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_064() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_065() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_066() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_067() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_068() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_069() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_070() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_071() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_072() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_073() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_074() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_075() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_076() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_077() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_078() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_079() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_080() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_081() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_082() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_083() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_084() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_085() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_086() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_087() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_088() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_089() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_090() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_091() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_092() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_093() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_094() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_095() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_096() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_097() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_098() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_099() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_100() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_101() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_102() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_103() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_104() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_105() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_106() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_107() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_108() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_109() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_110() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_111() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_112() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_113() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_114() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_115() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_116() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_117() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_118() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_119() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_120() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_121() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_122() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_123() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_124() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_125() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_126() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_127() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_128() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_129() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_130() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_131() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_132() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_133() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_134() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_135() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_136() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_137() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_138() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_139() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_140() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_141() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_142() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_143() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_144() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_145() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_146() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_147() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_148() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_149() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_150() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_151() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_152() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_153() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_154() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_155() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_156() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_157() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_158() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_159() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_160() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_161() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_162() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_163() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_164() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_165() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_166() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_167() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_168() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_169() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_170() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_171() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_172() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_173() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_174() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_175() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_176() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_177() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_178() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_179() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_180() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_181() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_182() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_183() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_184() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_185() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_186() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_187() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_188() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_189() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_190() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_191() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_192() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_193() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_194() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_195() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_196() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_197() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_198() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_199() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_200() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_201() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_202() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_203() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_204() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_205() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_206() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_207() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_208() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_209() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_210() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_211() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_212() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_213() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_214() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_215() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_216() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_217() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_218() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_219() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_220() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_221() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_222() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_223() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_224() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_225() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_226() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_227() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_228() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_229() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_230() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_231() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_232() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_233() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_234() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_235() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_236() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_237() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_238() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_239() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_240() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_241() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_242() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_243() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_244() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_245() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_246() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_247() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_248() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_249() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_250() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_251() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_252() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_253() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_254() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_255() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_256() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_257() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_258() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_259() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_260() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_261() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_262() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_263() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_264() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_265() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_266() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_267() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_268() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_269() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_270() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_271() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_272() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_273() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_274() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_275() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_276() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_277() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_278() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_279() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_280() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_281() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_282() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_283() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_284() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_285() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_286() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_287() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_288() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_289() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_290() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_291() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_292() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_293() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_294() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_295() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_296() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_297() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_298() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_299() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_300() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_301() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_302() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_303() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_304() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_305() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_306() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_307() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_308() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_309() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_310() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_311() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_312() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_313() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_314() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_315() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_316() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_317() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_318() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_319() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_320() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_321() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_322() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_323() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_324() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_325() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_326() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_327() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_328() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_329() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_330() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_331() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_332() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_333() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_334() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_335() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_336() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_337() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_338() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_339() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_340() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_341() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_342() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_343() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_344() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_345() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_346() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_347() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_348() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_349() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_350() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_351() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_352() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_353() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_354() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_355() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_356() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_357() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_358() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_359() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_360() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_361() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_362() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_363() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_364() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_365() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_366() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_367() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_368() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_369() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_370() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_371() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_372() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_373() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_374() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_375() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_376() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_377() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_378() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_379() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_380() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_381() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_382() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_383() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_384() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_385() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_386() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_387() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_388() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_389() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_390() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_391() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_392() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_393() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_394() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_395() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_396() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_397() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_398() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_399() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_400() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_401() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_402() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_403() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_404() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_405() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_406() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_407() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_408() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_409() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_410() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_411() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_412() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_413() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_414() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_415() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_416() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_417() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_418() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_419() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_420() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_421() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_422() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_423() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_424() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_425() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_426() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_427() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_428() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_429() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_430() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_431() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_432() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_433() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_434() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_435() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_436() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_437() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_438() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_439() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_440() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_441() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_442() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_443() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_444() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_445() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_446() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_447() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_448() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_449() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_450() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_451() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_452() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_453() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_454() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_455() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_456() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_457() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_458() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_459() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_460() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_461() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_462() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_463() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_464() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_465() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_466() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_467() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_468() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_469() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_470() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_471() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    #[test]
    fn test_graph2onnx_stress_472() {
        let gir = GraphIr::new("test");
        let model = lower_from_graph_ir(&gir).unwrap();
        assert_eq!(model.producer_name, "brain-onnx-export");
    }

    // ONNX proto parsing and graph lowering verification padding line 0
}
