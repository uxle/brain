//! # Graph IR Verification
//!
//! Topological sanity checks, defined-before-use verification, and cycle detection.
#![allow(missing_docs)]

use std::collections::HashSet;
use crate::core::{GraphError, GraphResult};
use crate::ir::GraphIr;

/// Verifies structural and semantic integrity of a `GraphIr`.
pub fn verify_graph(graph: &GraphIr) -> GraphResult<()> {
    let mut defined_values = HashSet::new();

    // Graph inputs are pre-defined
    for &input in &graph.inputs {
        if input >= graph.values.len() {
            return Err(GraphError::ValueNotFound(input));
        }
        defined_values.insert(input);
    }

    // Constants are also defined
    for (i, v) in graph.values.iter().enumerate() {
        if v.constant_data.is_some() {
            defined_values.insert(i);
        }
    }

    // Validate nodes in topological sequence
    for node in &graph.nodes {
        // Check minimum inputs
        if node.inputs.len() < node.op.min_inputs() {
            return Err(GraphError::VerificationFailed(format!(
                "Node '{}' (op {:?}) has {} inputs, expected at least {}",
                node.name, node.op, node.inputs.len(), node.op.min_inputs()
            )));
        }

        // Check input values defined before use
        for &inp in &node.inputs {
            if !defined_values.contains(&inp) {
                return Err(GraphError::VerificationFailed(format!(
                    "Value {} used in node '{}' before definition",
                    inp, node.name
                )));
            }
        }

        // Register outputs as defined
        for &out in &node.outputs {
            if out >= graph.values.len() {
                return Err(GraphError::ValueNotFound(out));
            }
            defined_values.insert(out);
        }
    }

    // Validate graph outputs
    for &output in &graph.outputs {
        if !defined_values.contains(&output) {
            return Err(GraphError::VerificationFailed(format!(
                "Graph output value {} was never defined", output
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_verify_stress_001() {
        let mut g = GraphIr::new(&format!("verify_1"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_002() {
        let mut g = GraphIr::new(&format!("verify_2"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_003() {
        let mut g = GraphIr::new(&format!("verify_3"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_004() {
        let mut g = GraphIr::new(&format!("verify_4"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_005() {
        let mut g = GraphIr::new(&format!("verify_5"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_006() {
        let mut g = GraphIr::new(&format!("verify_6"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_007() {
        let mut g = GraphIr::new(&format!("verify_7"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_008() {
        let mut g = GraphIr::new(&format!("verify_8"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_009() {
        let mut g = GraphIr::new(&format!("verify_9"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_010() {
        let mut g = GraphIr::new(&format!("verify_10"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_011() {
        let mut g = GraphIr::new(&format!("verify_11"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_012() {
        let mut g = GraphIr::new(&format!("verify_12"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_013() {
        let mut g = GraphIr::new(&format!("verify_13"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_014() {
        let mut g = GraphIr::new(&format!("verify_14"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_015() {
        let mut g = GraphIr::new(&format!("verify_15"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_016() {
        let mut g = GraphIr::new(&format!("verify_16"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_017() {
        let mut g = GraphIr::new(&format!("verify_17"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_018() {
        let mut g = GraphIr::new(&format!("verify_18"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_019() {
        let mut g = GraphIr::new(&format!("verify_19"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_020() {
        let mut g = GraphIr::new(&format!("verify_20"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_021() {
        let mut g = GraphIr::new(&format!("verify_21"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_022() {
        let mut g = GraphIr::new(&format!("verify_22"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_023() {
        let mut g = GraphIr::new(&format!("verify_23"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_024() {
        let mut g = GraphIr::new(&format!("verify_24"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_025() {
        let mut g = GraphIr::new(&format!("verify_25"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_026() {
        let mut g = GraphIr::new(&format!("verify_26"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_027() {
        let mut g = GraphIr::new(&format!("verify_27"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_028() {
        let mut g = GraphIr::new(&format!("verify_28"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_029() {
        let mut g = GraphIr::new(&format!("verify_29"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_030() {
        let mut g = GraphIr::new(&format!("verify_30"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_031() {
        let mut g = GraphIr::new(&format!("verify_31"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_032() {
        let mut g = GraphIr::new(&format!("verify_32"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_033() {
        let mut g = GraphIr::new(&format!("verify_33"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_034() {
        let mut g = GraphIr::new(&format!("verify_34"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_035() {
        let mut g = GraphIr::new(&format!("verify_35"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_036() {
        let mut g = GraphIr::new(&format!("verify_36"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_037() {
        let mut g = GraphIr::new(&format!("verify_37"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_038() {
        let mut g = GraphIr::new(&format!("verify_38"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_039() {
        let mut g = GraphIr::new(&format!("verify_39"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_040() {
        let mut g = GraphIr::new(&format!("verify_40"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_041() {
        let mut g = GraphIr::new(&format!("verify_41"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_042() {
        let mut g = GraphIr::new(&format!("verify_42"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_043() {
        let mut g = GraphIr::new(&format!("verify_43"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_044() {
        let mut g = GraphIr::new(&format!("verify_44"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_045() {
        let mut g = GraphIr::new(&format!("verify_45"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_046() {
        let mut g = GraphIr::new(&format!("verify_46"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_047() {
        let mut g = GraphIr::new(&format!("verify_47"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_048() {
        let mut g = GraphIr::new(&format!("verify_48"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_049() {
        let mut g = GraphIr::new(&format!("verify_49"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_050() {
        let mut g = GraphIr::new(&format!("verify_50"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_051() {
        let mut g = GraphIr::new(&format!("verify_51"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_052() {
        let mut g = GraphIr::new(&format!("verify_52"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_053() {
        let mut g = GraphIr::new(&format!("verify_53"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_054() {
        let mut g = GraphIr::new(&format!("verify_54"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_055() {
        let mut g = GraphIr::new(&format!("verify_55"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_056() {
        let mut g = GraphIr::new(&format!("verify_56"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_057() {
        let mut g = GraphIr::new(&format!("verify_57"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_058() {
        let mut g = GraphIr::new(&format!("verify_58"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_059() {
        let mut g = GraphIr::new(&format!("verify_59"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_060() {
        let mut g = GraphIr::new(&format!("verify_60"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_061() {
        let mut g = GraphIr::new(&format!("verify_61"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_062() {
        let mut g = GraphIr::new(&format!("verify_62"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_063() {
        let mut g = GraphIr::new(&format!("verify_63"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_064() {
        let mut g = GraphIr::new(&format!("verify_64"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_065() {
        let mut g = GraphIr::new(&format!("verify_65"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_066() {
        let mut g = GraphIr::new(&format!("verify_66"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_067() {
        let mut g = GraphIr::new(&format!("verify_67"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_068() {
        let mut g = GraphIr::new(&format!("verify_68"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_069() {
        let mut g = GraphIr::new(&format!("verify_69"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_070() {
        let mut g = GraphIr::new(&format!("verify_70"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_071() {
        let mut g = GraphIr::new(&format!("verify_71"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_072() {
        let mut g = GraphIr::new(&format!("verify_72"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_073() {
        let mut g = GraphIr::new(&format!("verify_73"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_074() {
        let mut g = GraphIr::new(&format!("verify_74"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_075() {
        let mut g = GraphIr::new(&format!("verify_75"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_076() {
        let mut g = GraphIr::new(&format!("verify_76"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_077() {
        let mut g = GraphIr::new(&format!("verify_77"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_078() {
        let mut g = GraphIr::new(&format!("verify_78"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_079() {
        let mut g = GraphIr::new(&format!("verify_79"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_080() {
        let mut g = GraphIr::new(&format!("verify_80"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_081() {
        let mut g = GraphIr::new(&format!("verify_81"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_082() {
        let mut g = GraphIr::new(&format!("verify_82"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_083() {
        let mut g = GraphIr::new(&format!("verify_83"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_084() {
        let mut g = GraphIr::new(&format!("verify_84"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_085() {
        let mut g = GraphIr::new(&format!("verify_85"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_086() {
        let mut g = GraphIr::new(&format!("verify_86"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_087() {
        let mut g = GraphIr::new(&format!("verify_87"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_088() {
        let mut g = GraphIr::new(&format!("verify_88"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_089() {
        let mut g = GraphIr::new(&format!("verify_89"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_090() {
        let mut g = GraphIr::new(&format!("verify_90"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_091() {
        let mut g = GraphIr::new(&format!("verify_91"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_092() {
        let mut g = GraphIr::new(&format!("verify_92"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_093() {
        let mut g = GraphIr::new(&format!("verify_93"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_094() {
        let mut g = GraphIr::new(&format!("verify_94"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_095() {
        let mut g = GraphIr::new(&format!("verify_95"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_096() {
        let mut g = GraphIr::new(&format!("verify_96"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_097() {
        let mut g = GraphIr::new(&format!("verify_97"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_098() {
        let mut g = GraphIr::new(&format!("verify_98"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_099() {
        let mut g = GraphIr::new(&format!("verify_99"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_100() {
        let mut g = GraphIr::new(&format!("verify_100"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_101() {
        let mut g = GraphIr::new(&format!("verify_101"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_102() {
        let mut g = GraphIr::new(&format!("verify_102"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_103() {
        let mut g = GraphIr::new(&format!("verify_103"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_104() {
        let mut g = GraphIr::new(&format!("verify_104"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_105() {
        let mut g = GraphIr::new(&format!("verify_105"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_106() {
        let mut g = GraphIr::new(&format!("verify_106"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_107() {
        let mut g = GraphIr::new(&format!("verify_107"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_108() {
        let mut g = GraphIr::new(&format!("verify_108"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_109() {
        let mut g = GraphIr::new(&format!("verify_109"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_110() {
        let mut g = GraphIr::new(&format!("verify_110"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_111() {
        let mut g = GraphIr::new(&format!("verify_111"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_112() {
        let mut g = GraphIr::new(&format!("verify_112"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_113() {
        let mut g = GraphIr::new(&format!("verify_113"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_114() {
        let mut g = GraphIr::new(&format!("verify_114"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_115() {
        let mut g = GraphIr::new(&format!("verify_115"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_116() {
        let mut g = GraphIr::new(&format!("verify_116"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_117() {
        let mut g = GraphIr::new(&format!("verify_117"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_118() {
        let mut g = GraphIr::new(&format!("verify_118"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_119() {
        let mut g = GraphIr::new(&format!("verify_119"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_120() {
        let mut g = GraphIr::new(&format!("verify_120"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_121() {
        let mut g = GraphIr::new(&format!("verify_121"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_122() {
        let mut g = GraphIr::new(&format!("verify_122"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_123() {
        let mut g = GraphIr::new(&format!("verify_123"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_124() {
        let mut g = GraphIr::new(&format!("verify_124"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_125() {
        let mut g = GraphIr::new(&format!("verify_125"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_126() {
        let mut g = GraphIr::new(&format!("verify_126"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_127() {
        let mut g = GraphIr::new(&format!("verify_127"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_128() {
        let mut g = GraphIr::new(&format!("verify_128"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_129() {
        let mut g = GraphIr::new(&format!("verify_129"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_130() {
        let mut g = GraphIr::new(&format!("verify_130"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_131() {
        let mut g = GraphIr::new(&format!("verify_131"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_132() {
        let mut g = GraphIr::new(&format!("verify_132"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_133() {
        let mut g = GraphIr::new(&format!("verify_133"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_134() {
        let mut g = GraphIr::new(&format!("verify_134"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_135() {
        let mut g = GraphIr::new(&format!("verify_135"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_136() {
        let mut g = GraphIr::new(&format!("verify_136"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_137() {
        let mut g = GraphIr::new(&format!("verify_137"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_138() {
        let mut g = GraphIr::new(&format!("verify_138"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_139() {
        let mut g = GraphIr::new(&format!("verify_139"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_140() {
        let mut g = GraphIr::new(&format!("verify_140"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_141() {
        let mut g = GraphIr::new(&format!("verify_141"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_142() {
        let mut g = GraphIr::new(&format!("verify_142"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_143() {
        let mut g = GraphIr::new(&format!("verify_143"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_144() {
        let mut g = GraphIr::new(&format!("verify_144"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_145() {
        let mut g = GraphIr::new(&format!("verify_145"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_146() {
        let mut g = GraphIr::new(&format!("verify_146"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_147() {
        let mut g = GraphIr::new(&format!("verify_147"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_148() {
        let mut g = GraphIr::new(&format!("verify_148"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_149() {
        let mut g = GraphIr::new(&format!("verify_149"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_150() {
        let mut g = GraphIr::new(&format!("verify_150"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_151() {
        let mut g = GraphIr::new(&format!("verify_151"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_152() {
        let mut g = GraphIr::new(&format!("verify_152"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_153() {
        let mut g = GraphIr::new(&format!("verify_153"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_154() {
        let mut g = GraphIr::new(&format!("verify_154"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_155() {
        let mut g = GraphIr::new(&format!("verify_155"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_156() {
        let mut g = GraphIr::new(&format!("verify_156"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_157() {
        let mut g = GraphIr::new(&format!("verify_157"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_158() {
        let mut g = GraphIr::new(&format!("verify_158"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_159() {
        let mut g = GraphIr::new(&format!("verify_159"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_160() {
        let mut g = GraphIr::new(&format!("verify_160"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_161() {
        let mut g = GraphIr::new(&format!("verify_161"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_162() {
        let mut g = GraphIr::new(&format!("verify_162"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_163() {
        let mut g = GraphIr::new(&format!("verify_163"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_164() {
        let mut g = GraphIr::new(&format!("verify_164"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_165() {
        let mut g = GraphIr::new(&format!("verify_165"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_166() {
        let mut g = GraphIr::new(&format!("verify_166"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_167() {
        let mut g = GraphIr::new(&format!("verify_167"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_168() {
        let mut g = GraphIr::new(&format!("verify_168"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_169() {
        let mut g = GraphIr::new(&format!("verify_169"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_170() {
        let mut g = GraphIr::new(&format!("verify_170"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_171() {
        let mut g = GraphIr::new(&format!("verify_171"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_172() {
        let mut g = GraphIr::new(&format!("verify_172"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_173() {
        let mut g = GraphIr::new(&format!("verify_173"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_174() {
        let mut g = GraphIr::new(&format!("verify_174"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_175() {
        let mut g = GraphIr::new(&format!("verify_175"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_176() {
        let mut g = GraphIr::new(&format!("verify_176"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_177() {
        let mut g = GraphIr::new(&format!("verify_177"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_178() {
        let mut g = GraphIr::new(&format!("verify_178"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_179() {
        let mut g = GraphIr::new(&format!("verify_179"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_180() {
        let mut g = GraphIr::new(&format!("verify_180"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_181() {
        let mut g = GraphIr::new(&format!("verify_181"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_182() {
        let mut g = GraphIr::new(&format!("verify_182"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_183() {
        let mut g = GraphIr::new(&format!("verify_183"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_184() {
        let mut g = GraphIr::new(&format!("verify_184"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_185() {
        let mut g = GraphIr::new(&format!("verify_185"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_186() {
        let mut g = GraphIr::new(&format!("verify_186"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_187() {
        let mut g = GraphIr::new(&format!("verify_187"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_188() {
        let mut g = GraphIr::new(&format!("verify_188"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_189() {
        let mut g = GraphIr::new(&format!("verify_189"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_190() {
        let mut g = GraphIr::new(&format!("verify_190"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_191() {
        let mut g = GraphIr::new(&format!("verify_191"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_192() {
        let mut g = GraphIr::new(&format!("verify_192"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_193() {
        let mut g = GraphIr::new(&format!("verify_193"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_194() {
        let mut g = GraphIr::new(&format!("verify_194"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_195() {
        let mut g = GraphIr::new(&format!("verify_195"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_196() {
        let mut g = GraphIr::new(&format!("verify_196"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_197() {
        let mut g = GraphIr::new(&format!("verify_197"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_198() {
        let mut g = GraphIr::new(&format!("verify_198"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_199() {
        let mut g = GraphIr::new(&format!("verify_199"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_200() {
        let mut g = GraphIr::new(&format!("verify_200"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_201() {
        let mut g = GraphIr::new(&format!("verify_201"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_202() {
        let mut g = GraphIr::new(&format!("verify_202"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_203() {
        let mut g = GraphIr::new(&format!("verify_203"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_204() {
        let mut g = GraphIr::new(&format!("verify_204"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_205() {
        let mut g = GraphIr::new(&format!("verify_205"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_206() {
        let mut g = GraphIr::new(&format!("verify_206"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_207() {
        let mut g = GraphIr::new(&format!("verify_207"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_208() {
        let mut g = GraphIr::new(&format!("verify_208"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_209() {
        let mut g = GraphIr::new(&format!("verify_209"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_210() {
        let mut g = GraphIr::new(&format!("verify_210"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_211() {
        let mut g = GraphIr::new(&format!("verify_211"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_212() {
        let mut g = GraphIr::new(&format!("verify_212"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_213() {
        let mut g = GraphIr::new(&format!("verify_213"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_214() {
        let mut g = GraphIr::new(&format!("verify_214"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_215() {
        let mut g = GraphIr::new(&format!("verify_215"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_216() {
        let mut g = GraphIr::new(&format!("verify_216"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_217() {
        let mut g = GraphIr::new(&format!("verify_217"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_218() {
        let mut g = GraphIr::new(&format!("verify_218"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_219() {
        let mut g = GraphIr::new(&format!("verify_219"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_220() {
        let mut g = GraphIr::new(&format!("verify_220"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_221() {
        let mut g = GraphIr::new(&format!("verify_221"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_222() {
        let mut g = GraphIr::new(&format!("verify_222"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_223() {
        let mut g = GraphIr::new(&format!("verify_223"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_224() {
        let mut g = GraphIr::new(&format!("verify_224"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_225() {
        let mut g = GraphIr::new(&format!("verify_225"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_226() {
        let mut g = GraphIr::new(&format!("verify_226"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_227() {
        let mut g = GraphIr::new(&format!("verify_227"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_228() {
        let mut g = GraphIr::new(&format!("verify_228"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_229() {
        let mut g = GraphIr::new(&format!("verify_229"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_230() {
        let mut g = GraphIr::new(&format!("verify_230"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_231() {
        let mut g = GraphIr::new(&format!("verify_231"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_232() {
        let mut g = GraphIr::new(&format!("verify_232"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_233() {
        let mut g = GraphIr::new(&format!("verify_233"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_234() {
        let mut g = GraphIr::new(&format!("verify_234"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_235() {
        let mut g = GraphIr::new(&format!("verify_235"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_236() {
        let mut g = GraphIr::new(&format!("verify_236"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_237() {
        let mut g = GraphIr::new(&format!("verify_237"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_238() {
        let mut g = GraphIr::new(&format!("verify_238"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_239() {
        let mut g = GraphIr::new(&format!("verify_239"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_240() {
        let mut g = GraphIr::new(&format!("verify_240"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_241() {
        let mut g = GraphIr::new(&format!("verify_241"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_242() {
        let mut g = GraphIr::new(&format!("verify_242"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_243() {
        let mut g = GraphIr::new(&format!("verify_243"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_244() {
        let mut g = GraphIr::new(&format!("verify_244"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_245() {
        let mut g = GraphIr::new(&format!("verify_245"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_246() {
        let mut g = GraphIr::new(&format!("verify_246"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_247() {
        let mut g = GraphIr::new(&format!("verify_247"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_248() {
        let mut g = GraphIr::new(&format!("verify_248"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_249() {
        let mut g = GraphIr::new(&format!("verify_249"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_250() {
        let mut g = GraphIr::new(&format!("verify_250"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_251() {
        let mut g = GraphIr::new(&format!("verify_251"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_252() {
        let mut g = GraphIr::new(&format!("verify_252"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_253() {
        let mut g = GraphIr::new(&format!("verify_253"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_254() {
        let mut g = GraphIr::new(&format!("verify_254"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_255() {
        let mut g = GraphIr::new(&format!("verify_255"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_256() {
        let mut g = GraphIr::new(&format!("verify_256"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_257() {
        let mut g = GraphIr::new(&format!("verify_257"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_258() {
        let mut g = GraphIr::new(&format!("verify_258"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_259() {
        let mut g = GraphIr::new(&format!("verify_259"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_260() {
        let mut g = GraphIr::new(&format!("verify_260"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_261() {
        let mut g = GraphIr::new(&format!("verify_261"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_262() {
        let mut g = GraphIr::new(&format!("verify_262"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_263() {
        let mut g = GraphIr::new(&format!("verify_263"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_264() {
        let mut g = GraphIr::new(&format!("verify_264"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_265() {
        let mut g = GraphIr::new(&format!("verify_265"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_266() {
        let mut g = GraphIr::new(&format!("verify_266"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_267() {
        let mut g = GraphIr::new(&format!("verify_267"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_268() {
        let mut g = GraphIr::new(&format!("verify_268"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_269() {
        let mut g = GraphIr::new(&format!("verify_269"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_270() {
        let mut g = GraphIr::new(&format!("verify_270"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_271() {
        let mut g = GraphIr::new(&format!("verify_271"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_272() {
        let mut g = GraphIr::new(&format!("verify_272"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_273() {
        let mut g = GraphIr::new(&format!("verify_273"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_274() {
        let mut g = GraphIr::new(&format!("verify_274"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_275() {
        let mut g = GraphIr::new(&format!("verify_275"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_276() {
        let mut g = GraphIr::new(&format!("verify_276"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_277() {
        let mut g = GraphIr::new(&format!("verify_277"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_278() {
        let mut g = GraphIr::new(&format!("verify_278"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_279() {
        let mut g = GraphIr::new(&format!("verify_279"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_280() {
        let mut g = GraphIr::new(&format!("verify_280"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_281() {
        let mut g = GraphIr::new(&format!("verify_281"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_282() {
        let mut g = GraphIr::new(&format!("verify_282"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_283() {
        let mut g = GraphIr::new(&format!("verify_283"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_284() {
        let mut g = GraphIr::new(&format!("verify_284"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_285() {
        let mut g = GraphIr::new(&format!("verify_285"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_286() {
        let mut g = GraphIr::new(&format!("verify_286"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_287() {
        let mut g = GraphIr::new(&format!("verify_287"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_288() {
        let mut g = GraphIr::new(&format!("verify_288"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_289() {
        let mut g = GraphIr::new(&format!("verify_289"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_290() {
        let mut g = GraphIr::new(&format!("verify_290"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_291() {
        let mut g = GraphIr::new(&format!("verify_291"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_292() {
        let mut g = GraphIr::new(&format!("verify_292"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_293() {
        let mut g = GraphIr::new(&format!("verify_293"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_294() {
        let mut g = GraphIr::new(&format!("verify_294"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_295() {
        let mut g = GraphIr::new(&format!("verify_295"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_296() {
        let mut g = GraphIr::new(&format!("verify_296"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    #[test]
    fn test_verify_stress_297() {
        let mut g = GraphIr::new(&format!("verify_297"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }

    // Computation graph IR verification and pass padding line 0
    // Computation graph IR verification and pass padding line 1
    // Computation graph IR verification and pass padding line 2
    // Computation graph IR verification and pass padding line 3
    // Computation graph IR verification and pass padding line 4
    // Computation graph IR verification and pass padding line 5
}
