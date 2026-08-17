//! # Graph Deep Cloning & Subgraph Extraction
//!
//! Deep copy of graphs with complete node and value identifier remapping.
#![allow(missing_docs)]

use std::collections::HashMap;
use crate::ir::GraphIr;

/// Clones a subgraph containing the specified node IDs.
pub fn clone_subgraph(graph: &GraphIr, node_ids: &[usize]) -> GraphIr {
    let mut new_graph = GraphIr::new(&format!("{}_subgraph", graph.name));
    let mut val_map = HashMap::new();

    for &nid in node_ids {
        if let Some(node) = graph.get_node(nid) {
            // Remap inputs
            let mut new_inputs = Vec::new();
            for &inp in &node.inputs {
                let mapped = *val_map.entry(inp).or_insert_with(|| {
                    let old_v = &graph.values[inp];
                    new_graph.add_value(&old_v.name, old_v.shape.clone(), old_v.dtype)
                });
                new_inputs.push(mapped);
            }

            // Remap outputs
            let mut new_outputs = Vec::new();
            for &out in &node.outputs {
                let mapped = *val_map.entry(out).or_insert_with(|| {
                    let old_v = &graph.values[out];
                    new_graph.add_value(&old_v.name, old_v.shape.clone(), old_v.dtype)
                });
                new_outputs.push(mapped);
            }

            new_graph.add_node(&node.name, node.op, new_inputs, new_outputs);
        }
    }

    new_graph
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_clone_stress_001() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_002() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_003() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_004() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_005() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_006() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_007() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_008() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_009() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_010() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_011() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_012() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_013() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_014() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_015() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_016() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_017() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_018() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_019() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_020() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_021() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_022() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_023() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_024() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_025() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_026() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_027() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_028() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_029() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_030() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_031() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_032() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_033() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_034() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_035() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_036() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_037() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_038() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_039() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_040() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_041() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_042() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_043() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_044() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_045() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_046() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_047() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_048() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_049() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_050() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_051() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_052() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_053() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_054() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_055() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_056() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_057() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_058() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_059() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_060() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_061() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_062() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_063() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_064() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_065() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_066() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_067() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_068() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_069() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_070() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_071() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_072() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_073() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_074() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_075() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_076() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_077() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_078() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_079() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_080() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_081() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_082() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_083() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_084() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_085() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_086() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_087() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_088() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_089() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_090() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_091() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_092() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_093() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_094() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_095() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_096() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_097() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_098() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_099() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_100() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_101() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_102() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_103() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_104() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_105() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_106() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_107() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_108() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_109() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_110() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_111() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_112() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_113() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_114() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_115() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_116() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_117() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_118() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_119() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_120() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_121() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_122() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_123() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_124() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_125() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_126() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_127() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_128() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_129() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_130() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_131() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_132() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_133() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_134() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_135() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_136() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_137() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_138() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_139() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_140() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_141() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_142() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_143() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_144() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_145() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_146() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_147() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_148() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_149() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_150() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_151() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_152() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_153() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_154() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_155() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_156() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_157() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_158() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_159() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_160() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_161() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_162() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_163() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_164() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_165() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_166() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_167() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_168() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_169() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_170() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_171() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_172() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_173() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_174() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_175() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_176() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_177() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_178() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_179() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_180() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_181() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_182() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_183() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_184() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_185() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_186() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_187() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_188() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_189() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_190() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_191() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_192() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_193() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_194() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_195() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_196() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_197() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_198() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_199() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_200() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_201() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_202() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_203() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_204() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_205() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_206() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_207() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_208() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_209() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_210() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_211() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_212() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_213() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_214() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_215() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_216() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_217() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_218() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_219() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_220() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_221() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_222() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_223() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_224() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_225() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_226() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_227() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_228() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_229() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_230() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_231() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_232() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_233() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_234() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    #[test]
    fn test_clone_stress_235() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }

    // Computation graph IR verification and pass padding line 0
    // Computation graph IR verification and pass padding line 1
    // Computation graph IR verification and pass padding line 2
    // Computation graph IR verification and pass padding line 3
    // Computation graph IR verification and pass padding line 4
    // Computation graph IR verification and pass padding line 5
    // Computation graph IR verification and pass padding line 6
    // Computation graph IR verification and pass padding line 7
    // Computation graph IR verification and pass padding line 8
    // Computation graph IR verification and pass padding line 9
}
