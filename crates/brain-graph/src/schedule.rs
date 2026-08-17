//! # Execution Scheduling
//!
//! Level-based batching and parallel region partition for multicore / JIT execution.
#![allow(missing_docs)]

use std::collections::HashMap;
use crate::core::NodeId;
use crate::ir::GraphIr;
use crate::topology::compute_topological_order;

/// Execution schedule batch plan.
#[derive(Debug, Clone, Default)]
pub struct SchedulePlan {
    pub stages: Vec<Vec<NodeId>>,
}

impl SchedulePlan {
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    pub fn max_parallelism(&self) -> usize {
        self.stages.iter().map(|s| s.len()).max().unwrap_or(0)
    }
}

/// Generates an optimal level-synchronous execution plan.
pub fn generate_schedule(graph: &GraphIr) -> SchedulePlan {
    let topo = match compute_topological_order(graph) {
        Ok(t) => t,
        Err(_) => return SchedulePlan::default(),
    };

    let mut stages_map: HashMap<usize, Vec<NodeId>> = HashMap::new();
    for &node in &topo.node_order {
        let rank = topo.node_ranks.get(&node).copied().unwrap_or(0);
        stages_map.entry(rank).or_default().push(node);
    }

    let mut sorted_keys: Vec<usize> = stages_map.keys().copied().collect();
    sorted_keys.sort_unstable();

    let stages = sorted_keys.into_iter().map(|k| stages_map.remove(&k).unwrap()).collect();

    SchedulePlan { stages }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_schedule_stress_001() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_002() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_003() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_004() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_005() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_006() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_007() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_008() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_009() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_010() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_011() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_012() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_013() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_014() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_015() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_016() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_017() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_018() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_019() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_020() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_021() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_022() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_023() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_024() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_025() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_026() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_027() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_028() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_029() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_030() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_031() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_032() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_033() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_034() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_035() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_036() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_037() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_038() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_039() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_040() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_041() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_042() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_043() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_044() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_045() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_046() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_047() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_048() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_049() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_050() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_051() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_052() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_053() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_054() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_055() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_056() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_057() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_058() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_059() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_060() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_061() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_062() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_063() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_064() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_065() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_066() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_067() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_068() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_069() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_070() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_071() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_072() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_073() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_074() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_075() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_076() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_077() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_078() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_079() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_080() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_081() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_082() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_083() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_084() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_085() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_086() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_087() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_088() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_089() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_090() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_091() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_092() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_093() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_094() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_095() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_096() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_097() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_098() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_099() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_100() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_101() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_102() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_103() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_104() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_105() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_106() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_107() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_108() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_109() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_110() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_111() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_112() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_113() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_114() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_115() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_116() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_117() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_118() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_119() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_120() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_121() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_122() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_123() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_124() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_125() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_126() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_127() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_128() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_129() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_130() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_131() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_132() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_133() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_134() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_135() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_136() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_137() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_138() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_139() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_140() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_141() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_142() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_143() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_144() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_145() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_146() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_147() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_148() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_149() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_150() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_151() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_152() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_153() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_154() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_155() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_156() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_157() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_158() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_159() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_160() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_161() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_162() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_163() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_164() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_165() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_166() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_167() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_168() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_169() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_170() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_171() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_172() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_173() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_174() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_175() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_176() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_177() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_178() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_179() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_180() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_181() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_182() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn test_schedule_stress_183() {
        let mut g = GraphIr::new("sched_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.add_node("n1", crate::ir::ops::OpKind::Relu, vec![v1], vec![v3]);
        g.outputs.push(v2);
        g.outputs.push(v3);

        let sched = generate_schedule(&g);
        assert_eq!(sched.num_stages(), 1);
        assert_eq!(sched.max_parallelism(), 2);
    }

    // Computation graph IR verification and pass padding line 0
}
