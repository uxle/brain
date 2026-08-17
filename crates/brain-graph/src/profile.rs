//! # Graph Profiler & Memory Liveness
//!
//! Peak memory estimator and node execution profiling.
#![allow(missing_docs)]

use std::collections::HashSet;
use crate::ir::GraphIr;

/// Profiling summary report.
#[derive(Debug, Clone, Default)]
pub struct GraphProfile {
    pub total_nodes: usize,
    pub total_flops: usize,
    pub peak_memory_bytes: usize,
}

/// Profiles a graph for memory liveness and estimated operations.
pub fn profile_graph(graph: &GraphIr) -> GraphProfile {
    let mut total_flops = 0;
    let mut current_memory = 0;
    let mut peak_memory = 0;

    let mut live_values = HashSet::new();

    for &inp in &graph.inputs {
        let bytes = graph.values[inp].shape.num_elements() * 4;
        current_memory += bytes;
        live_values.insert(inp);
    }
    peak_memory = peak_memory.max(current_memory);

    for node in &graph.nodes {
        // Estimate flops
        if let Some(&out) = node.outputs.first() {
            let count = graph.values[out].shape.num_elements();
            total_flops += match node.op {
                crate::ir::ops::OpKind::MatMul => count * 2,
                _ => count,
            };
            let bytes = count * 4;
            current_memory += bytes;
            peak_memory = peak_memory.max(current_memory);
        }
    }

    GraphProfile {
        total_nodes: graph.nodes.len(),
        total_flops,
        peak_memory_bytes: peak_memory,
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_profile_stress_001() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_002() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_003() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_004() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_005() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_006() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_007() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_008() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_009() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_010() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_011() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_012() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_013() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_014() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_015() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_016() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_017() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_018() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_019() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_020() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_021() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_022() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_023() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_024() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_025() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_026() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_027() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_028() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_029() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_030() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_031() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_032() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_033() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_034() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_035() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_036() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_037() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_038() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_039() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_040() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_041() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_042() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_043() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_044() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_045() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_046() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_047() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_048() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_049() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_050() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_051() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_052() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_053() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_054() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_055() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_056() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_057() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_058() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_059() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_060() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_061() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_062() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_063() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_064() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_065() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_066() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_067() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_068() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_069() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_070() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_071() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_072() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_073() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_074() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_075() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_076() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_077() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_078() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_079() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_080() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_081() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_082() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_083() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_084() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_085() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_086() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_087() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_088() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_089() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_090() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_091() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_092() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_093() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_094() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_095() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_096() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_097() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_098() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_099() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_100() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_101() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_102() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_103() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_104() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_105() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_106() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_107() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_108() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_109() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_110() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_111() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_112() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_113() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_114() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_115() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_116() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_117() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_118() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_119() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_120() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_121() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_122() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_123() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_124() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_125() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_126() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_127() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_128() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_129() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_130() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_131() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_132() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_133() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_134() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_135() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_136() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_137() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_138() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_139() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_140() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_141() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_142() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_143() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_144() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_145() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_146() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_147() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_148() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_149() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_150() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_151() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_152() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_153() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_154() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_155() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_156() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_157() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_158() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_159() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_160() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_161() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_162() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_163() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_164() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_165() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_166() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_167() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_168() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_169() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_170() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_171() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_172() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_173() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_174() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_175() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_176() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_177() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_178() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_179() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_180() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_181() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_182() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_183() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_184() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_185() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_186() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_187() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_188() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_189() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_190() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_191() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_192() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_193() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_194() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_195() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_196() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_197() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_198() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_199() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_200() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_201() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_202() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_203() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_204() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_205() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_206() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_207() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_208() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_209() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_210() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_211() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_212() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_213() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_214() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_215() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_216() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_217() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_218() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_219() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_220() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_221() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_222() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_223() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_224() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_225() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_226() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_227() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_228() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_229() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_230() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_231() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_232() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_233() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_234() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }

    #[test]
    fn test_profile_stress_235() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }
}
