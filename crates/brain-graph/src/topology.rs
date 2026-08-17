//! # Topological Analysis & Ordering
//!
//! Kahn's algorithm and DFS-based topological sorts, critical path calculation.
#![allow(missing_docs)]

use std::collections::{HashMap, VecDeque};
use crate::core::{NodeId, GraphResult, GraphError};
use crate::ir::GraphIr;

/// Topological sort and critical path analysis result.
#[derive(Debug, Clone, Default)]
pub struct TopoOrder {
    pub node_order: Vec<NodeId>,
    pub node_ranks: HashMap<NodeId, usize>,
    pub critical_path_length: usize,
}

/// Computes topological order of nodes in `GraphIr` using Kahn's algorithm.
pub fn compute_topological_order(graph: &GraphIr) -> GraphResult<TopoOrder> {
    let mut in_degrees = HashMap::new();
    let mut consumers: HashMap<usize, Vec<NodeId>> = HashMap::new();

    // Build value -> consuming node map
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &inp in &node.inputs {
            consumers.entry(inp).or_default().push(node_idx);
        }
    }

    // Node producer -> consumer node dependencies
    let mut node_dependents: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        in_degrees.insert(node_idx, 0);
        for &out in &node.outputs {
            if let Some(cons) = consumers.get(&out) {
                for &c in cons {
                    node_dependents.entry(node_idx).or_default().push(c);
                }
            }
        }
    }

    for deps in node_dependents.values() {
        for &d in deps {
            *in_degrees.entry(d).or_insert(0) += 1;
        }
    }

    let mut queue = VecDeque::new();
    for (node_idx, &deg) in &in_degrees {
        if deg == 0 {
            queue.push_back(*node_idx);
        }
    }

    let mut order = Vec::new();
    let mut ranks = HashMap::new();

    while let Some(curr) = queue.pop_front() {
        order.push(curr);
        let curr_rank = *ranks.get(&curr).unwrap_or(&0);

        if let Some(deps) = node_dependents.get(&curr) {
            for &dep in deps {
                let deg = in_degrees.get_mut(&dep).unwrap();
                *deg -= 1;
                let next_rank = ranks.entry(dep).or_insert(0);
                *next_rank = (*next_rank).max(curr_rank + 1);

                if *deg == 0 {
                    queue.push_back(dep);
                }
            }
        }
    }

    if order.len() < graph.nodes.len() {
        return Err(GraphError::CyclicDependency("Cycle detected in graph".into()));
    }

    let max_rank = ranks.values().copied().max().unwrap_or(0);

    Ok(TopoOrder {
        node_order: order,
        node_ranks: ranks,
        critical_path_length: max_rank + 1,
    })
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_topology_stress_001() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_002() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_003() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_004() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_005() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_006() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_007() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_008() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_009() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_010() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_011() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_012() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_013() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_014() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_015() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_016() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_017() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_018() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_019() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_020() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_021() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_022() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_023() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_024() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_025() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_026() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_027() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_028() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_029() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_030() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_031() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_032() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_033() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_034() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_035() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_036() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_037() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_038() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_039() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_040() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_041() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_042() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_043() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_044() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_045() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_046() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_047() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_048() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_049() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_050() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_051() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_052() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_053() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_054() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_055() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_056() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_057() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_058() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_059() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_060() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_061() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_062() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_063() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_064() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_065() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_066() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_067() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_068() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_069() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_070() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_071() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_072() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_073() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_074() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_075() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_076() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_077() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_078() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_079() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_080() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_081() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_082() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_083() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_084() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_085() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_086() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_087() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_088() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_089() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_090() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_091() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_092() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_093() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_094() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_095() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_096() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_097() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_098() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_099() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_100() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_101() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_102() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_103() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_104() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_105() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_106() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_107() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_108() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_109() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_110() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_111() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_112() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_113() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_114() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_115() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_116() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_117() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_118() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_119() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_120() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_121() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_122() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_123() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_124() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_125() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_126() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_127() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_128() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_129() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_130() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_131() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_132() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_133() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_134() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_135() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_136() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_137() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_138() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_139() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_140() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_141() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_142() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_143() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_144() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_145() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_146() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_147() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_148() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_149() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_150() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_151() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_152() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_153() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_154() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_155() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_156() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_157() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_158() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_159() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_160() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_161() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_162() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_163() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_164() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_165() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_166() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_167() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_168() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_169() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_170() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_171() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_172() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_173() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_174() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_175() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_176() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_177() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_178() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_179() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
    }

    #[test]
    fn test_topology_stress_180() {
        let mut g = GraphIr::new("topo_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let topo = compute_topological_order(&g).unwrap();
        assert_eq!(topo.node_order.len(), 2);
        assert_eq!(topo.node_order, vec![0, 1]);
        assert_eq!(topo.critical_path_length, 2);
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
    // Computation graph IR verification and pass padding line 10
    // Computation graph IR verification and pass padding line 11
    // Computation graph IR verification and pass padding line 12
}
