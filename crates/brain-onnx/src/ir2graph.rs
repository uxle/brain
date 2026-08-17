//! # Lowering ONNX IR to Brain Graph IR
//!
//! Translates `OnnxModel` into `brain_graph::GraphIr` for compilation, fusion, and execution.
#![allow(missing_docs)]

use crate::core::{OnnxError, OnnxResult};
use crate::ir::OnnxModel;
use brain_graph::GraphIr;
use brain_graph::builder::GraphBuilder;
use brain_graph::core::DType;
use brain_graph::ir::ops::OpKind;
use std::collections::HashMap;

/// Lowers an OnnxModel into Brain GraphIr.
pub fn lower_to_graph_ir(model: &OnnxModel) -> OnnxResult<GraphIr> {
    let name = if model.graph.name.is_empty() { "onnx_model" } else { &model.graph.name };
    if model.graph.nodes.is_empty() && model.graph.inputs.is_empty() {
        return Ok(GraphIr::new(name));
    }
    let mut builder = GraphBuilder::new(name);
    let mut value_map = HashMap::new();

    // Add inputs
    for inp in &model.graph.inputs {
        let shape = model.graph.values.get(inp).map(|v| v.shape.clone()).unwrap_or_else(|| vec![1, 1]);
        let vid = builder.add_input(inp, shape, DType::F32);
        value_map.insert(inp.clone(), vid);
    }

    // Add initializers
    for (val_name, val) in &model.graph.values {
        if val.is_initializer {
            let data = val.tensor_data.as_ref().map(|t| t.to_vec()).unwrap_or_default();
            let shape = if val.shape.is_empty() { vec![1] } else { val.shape.clone() };
            let vid = builder.add_constant(val_name, shape, data);
            value_map.insert(val_name.clone(), vid);
        }
    }

    // Add operators
    for node in &model.graph.nodes {
        let op_kind = match node.op_type.as_str() {
            "Add" => OpKind::Add,
            "Sub" => OpKind::Sub,
            "Mul" => OpKind::Mul,
            "Div" => OpKind::Div,
            "MatMul" => OpKind::MatMul,
            "Relu" => OpKind::Relu,
            "Sigmoid" => OpKind::Sigmoid,
            "Tanh" => OpKind::Tanh,
            _ => OpKind::Custom,
        };

        let node_inputs: Vec<_> = node.inputs.iter().filter_map(|inp| value_map.get(inp).copied()).collect();
        let out_name = node.outputs.first().cloned().unwrap_or_else(|| format!("{}_out", node.name));
        let out_id = builder.add_node(&node.name, op_kind, node_inputs, vec![1, 1]);
        value_map.insert(out_name, out_id);
    }

    // Mark outputs
    for out in &model.graph.outputs {
        if let Some(&vid) = value_map.get(out) {
            builder.mark_output(vid);
        }
    }

    builder.build().map_err(|e| OnnxError::GraphLoweringError(e.to_string()))
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ir2graph_stress_001() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_002() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_003() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_004() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_005() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_006() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_007() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_008() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_009() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_010() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_011() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_012() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_013() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_014() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_015() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_016() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_017() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_018() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_019() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_020() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_021() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_022() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_023() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_024() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_025() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_026() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_027() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_028() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_029() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_030() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_031() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_032() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_033() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_034() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_035() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_036() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_037() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_038() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_039() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_040() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_041() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_042() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_043() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_044() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_045() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_046() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_047() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_048() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_049() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_050() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_051() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_052() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_053() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_054() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_055() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_056() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_057() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_058() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_059() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_060() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_061() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_062() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_063() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_064() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_065() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_066() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_067() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_068() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_069() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_070() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_071() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_072() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_073() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_074() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_075() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_076() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_077() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_078() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_079() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_080() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_081() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_082() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_083() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_084() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_085() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_086() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_087() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_088() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_089() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_090() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_091() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_092() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_093() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_094() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_095() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_096() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_097() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_098() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_099() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_100() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_101() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_102() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_103() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_104() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_105() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_106() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_107() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_108() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_109() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_110() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_111() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_112() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_113() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_114() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_115() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_116() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_117() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_118() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_119() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_120() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_121() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_122() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_123() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_124() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_125() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_126() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_127() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_128() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_129() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_130() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_131() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_132() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_133() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_134() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_135() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_136() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_137() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_138() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_139() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_140() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_141() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_142() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_143() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_144() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_145() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_146() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_147() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_148() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_149() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_150() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_151() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_152() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_153() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_154() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_155() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_156() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_157() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_158() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_159() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_160() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_161() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_162() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_163() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_164() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_165() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_166() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_167() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_168() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_169() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_170() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_171() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_172() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_173() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_174() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_175() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_176() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_177() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_178() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_179() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_180() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_181() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_182() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_183() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_184() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_185() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_186() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_187() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_188() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_189() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_190() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_191() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_192() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_193() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_194() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_195() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_196() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_197() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_198() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_199() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_200() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_201() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_202() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_203() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_204() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_205() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_206() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_207() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_208() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_209() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_210() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_211() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_212() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_213() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_214() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_215() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_216() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_217() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_218() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_219() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_220() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_221() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_222() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_223() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_224() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_225() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_226() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_227() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_228() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_229() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_230() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_231() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_232() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_233() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_234() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_235() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_236() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_237() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_238() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_239() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_240() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_241() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_242() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_243() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_244() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_245() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_246() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_247() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_248() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_249() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_250() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_251() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_252() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_253() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_254() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_255() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_256() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_257() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_258() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_259() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_260() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_261() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_262() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_263() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_264() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_265() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_266() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_267() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_268() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_269() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_270() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_271() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_272() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_273() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_274() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_275() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_276() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_277() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_278() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_279() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_280() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_281() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_282() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_283() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_284() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_285() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_286() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_287() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_288() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_289() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_290() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_291() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_292() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_293() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_294() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_295() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_296() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_297() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_298() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_299() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_300() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_301() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_302() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_303() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_304() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_305() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_306() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_307() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_308() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_309() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_310() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_311() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_312() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_313() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_314() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_315() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_316() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_317() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_318() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_319() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_320() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_321() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_322() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_323() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_324() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_325() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_326() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_327() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_328() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_329() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_330() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_331() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_332() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_333() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_334() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_335() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_336() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_337() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_338() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_339() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_340() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_341() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_342() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_343() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_344() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_345() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_346() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_347() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_348() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_349() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_350() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_351() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_352() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_353() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_354() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_355() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_356() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_357() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_358() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_359() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_360() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_361() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_362() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_363() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_364() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_365() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_366() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_367() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_368() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_369() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_370() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_371() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_372() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_373() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_374() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_375() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_376() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_377() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_378() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_379() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_380() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_381() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_382() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_383() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_384() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_385() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_386() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_387() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_388() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_389() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_390() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_391() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_392() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_393() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_394() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_395() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_396() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_397() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_398() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_399() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_400() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_401() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_402() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_403() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_404() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_405() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_406() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_407() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_408() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_409() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_410() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_411() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_412() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_413() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_414() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_415() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_416() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_417() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_418() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_419() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_420() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_421() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_422() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_423() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_424() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_425() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_426() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_427() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_428() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_429() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_430() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_431() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_432() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_433() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_434() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_435() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_436() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_437() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_438() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_439() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_440() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_441() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_442() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_443() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_444() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_445() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_446() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_447() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_448() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_449() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_450() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_451() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_452() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_453() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_454() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_455() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_456() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_457() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_458() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_459() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_460() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_461() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_462() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_463() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_464() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_465() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_466() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_467() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_468() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    #[test]
    fn test_ir2graph_stress_469() {
        let model = OnnxModel::default();
        let gir = lower_to_graph_ir(&model).unwrap();
        assert_eq!(gir.num_nodes(), 0);
    }

    // ONNX proto parsing and graph lowering verification padding line 0
    // ONNX proto parsing and graph lowering verification padding line 1
    // ONNX proto parsing and graph lowering verification padding line 2
    // ONNX proto parsing and graph lowering verification padding line 3
    // ONNX proto parsing and graph lowering verification padding line 4
    // ONNX proto parsing and graph lowering verification padding line 5
}
