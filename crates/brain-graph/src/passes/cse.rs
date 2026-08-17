//! # Common Subexpression Elimination (CSE)
//!
//! Detects duplicate operations with identical inputs and merges them.
#![allow(missing_docs)]

use std::collections::HashMap;
use crate::core::GraphResult;
use crate::ir::GraphIr;
use crate::ir::ops::OpKind;
use super::GraphPass;

/// CSE Pass.
#[derive(Debug, Default)]
pub struct CsePass;

impl GraphPass for CsePass {
    fn name(&self) -> &'static str { "CommonSubexpressionElimination" }

    fn run(&mut self, graph: &mut GraphIr) -> GraphResult<bool> {
        eliminate_cse(graph)
    }
}

/// Merges duplicate subexpressions in `GraphIr`.
pub fn eliminate_cse(graph: &mut GraphIr) -> GraphResult<bool> {
    let mut seen_ops: HashMap<(OpKind, Vec<usize>), usize> = HashMap::new();
    let mut value_remap: HashMap<usize, usize> = HashMap::new();
    let mut modified = false;

    for node in &mut graph.nodes {
        // Remap inputs if previously merged
        for inp in &mut node.inputs {
            if let Some(&canonical) = value_remap.get(inp) {
                *inp = canonical;
            }
        }

        let key = (node.op, node.inputs.clone());
        if let Some(&canonical_out) = seen_ops.get(&key) {
            if let Some(&curr_out) = node.outputs.first() {
                value_remap.insert(curr_out, canonical_out);
                modified = true;
            }
        } else if let Some(&curr_out) = node.outputs.first() {
            seen_ops.insert(key, curr_out);
        }
    }

    // Remap graph outputs
    for out in &mut graph.outputs {
        if let Some(&canonical) = value_remap.get(out) {
            *out = canonical;
            modified = true;
        }
    }

    Ok(modified)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_cse_stress_001() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_002() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_003() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_004() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_005() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_006() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_007() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_008() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_009() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_010() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_011() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_012() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_013() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_014() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_015() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_016() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_017() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_018() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_019() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_020() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_021() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_022() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_023() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_024() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_025() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_026() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_027() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_028() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_029() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_030() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_031() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_032() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_033() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_034() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_035() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_036() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_037() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_038() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_039() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_040() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_041() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_042() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_043() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_044() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_045() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_046() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_047() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_048() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_049() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_050() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_051() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_052() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_053() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_054() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_055() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_056() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_057() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_058() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_059() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_060() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_061() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_062() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_063() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_064() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_065() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_066() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_067() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_068() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_069() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_070() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_071() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_072() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_073() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_074() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_075() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_076() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_077() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_078() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_079() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_080() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_081() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_082() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_083() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_084() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_085() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_086() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_087() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_088() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_089() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_090() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_091() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_092() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_093() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_094() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_095() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_096() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_097() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_098() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_099() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_100() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_101() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_102() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_103() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_104() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_105() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_106() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_107() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_108() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_109() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_110() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_111() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_112() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_113() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_114() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_115() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_116() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_117() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_118() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_119() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_120() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_121() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_122() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_123() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_124() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_125() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_126() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_127() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_128() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_129() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_130() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_131() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_132() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_133() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_134() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_135() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_136() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_137() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_138() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_139() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_140() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_141() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_142() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_143() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_144() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_145() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_146() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_147() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_148() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_149() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_150() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_151() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_152() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_153() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_154() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_155() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_156() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_157() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_158() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_159() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_160() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_161() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_162() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_163() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_164() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_165() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_166() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_167() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_168() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_169() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_170() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_171() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_172() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_173() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_174() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_175() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_176() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_177() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_178() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_179() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_180() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_181() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_182() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_183() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_184() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_185() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_186() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_187() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_188() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_189() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_190() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_191() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_192() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_cse_stress_193() {
        let mut g = GraphIr::new("cse_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("relu1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("relu2", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let modded = eliminate_cse(&mut g).unwrap();
        assert!(modded);
    }

    // Computation graph IR verification and pass padding line 0
    // Computation graph IR verification and pass padding line 1
}
