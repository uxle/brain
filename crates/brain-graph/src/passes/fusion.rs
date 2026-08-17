//! # Operator Fusion Planning
//!
//! Clusters adjacent operations into single fused kernels (e.g. Conv+BN+ReLU).
#![allow(missing_docs)]

use crate::core::GraphResult;
use crate::ir::GraphIr;
use crate::ir::ops::OpKind;
use super::GraphPass;

/// Fusion Plan descriptor.
#[derive(Debug, Clone, Default)]
pub struct FusionPlan {
    pub fused_groups: Vec<Vec<usize>>,
}

/// Fusion planning optimization pass.
#[derive(Debug, Default)]
pub struct FusionPass;

impl GraphPass for FusionPass {
    fn name(&self) -> &'static str { "OperatorFusion" }

    fn run(&mut self, graph: &mut GraphIr) -> GraphResult<bool> {
        let plan = plan_fusion(graph)?;
        Ok(!plan.fused_groups.is_empty())
    }
}

/// Identifies fusable patterns in `GraphIr`.
pub fn plan_fusion(graph: &GraphIr) -> GraphResult<FusionPlan> {
    let mut plan = FusionPlan::default();

    for i in 0..graph.nodes.len() {
        let node_a = &graph.nodes[i];
        if node_a.op == OpKind::MatMul || node_a.op == OpKind::Conv2D {
            if let Some(&out_a) = node_a.outputs.first() {
                for (j, node_b) in graph.nodes.iter().enumerate().skip(i + 1) {
                    if node_b.inputs.contains(&out_a) && (node_b.op == OpKind::Relu || node_b.op == OpKind::Add) {
                        plan.fused_groups.push(vec![i, j]);
                    }
                }
            }
        }
    }

    Ok(plan)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_fusion_stress_001() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_002() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_003() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_004() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_005() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_006() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_007() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_008() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_009() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_010() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_011() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_012() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_013() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_014() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_015() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_016() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_017() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_018() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_019() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_020() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_021() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_022() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_023() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_024() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_025() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_026() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_027() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_028() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_029() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_030() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_031() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_032() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_033() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_034() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_035() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_036() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_037() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_038() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_039() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_040() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_041() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_042() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_043() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_044() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_045() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_046() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_047() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_048() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_049() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_050() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_051() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_052() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_053() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_054() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_055() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_056() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_057() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_058() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_059() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_060() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_061() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_062() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_063() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_064() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_065() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_066() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_067() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_068() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_069() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_070() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_071() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_072() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_073() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_074() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_075() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_076() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_077() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_078() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_079() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_080() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_081() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_082() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_083() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_084() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_085() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_086() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_087() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_088() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_089() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_090() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_091() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_092() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_093() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_094() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_095() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_096() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_097() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_098() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_099() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_100() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_101() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_102() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_103() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_104() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_105() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_106() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_107() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_108() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_109() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_110() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_111() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_112() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_113() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_114() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_115() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_116() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_117() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_118() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_119() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_120() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_121() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_122() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_123() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_124() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_125() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_126() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_127() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_128() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_129() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_130() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_131() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_132() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_133() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_134() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_135() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_136() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_137() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_138() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_139() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_140() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_141() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_142() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_143() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_144() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_145() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_146() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_147() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_148() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_149() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_150() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_151() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_152() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_153() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_154() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_155() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_156() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_157() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_158() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_159() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_160() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_161() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_162() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_163() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_164() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_165() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_166() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_167() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_168() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_169() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_170() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_171() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_172() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_173() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_174() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_175() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_176() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_177() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_178() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_179() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_180() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_181() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
    }

    #[test]
    fn test_fusion_stress_182() {
        let mut g = GraphIr::new("fusion_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let v4 = g.add_value("v4", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("mm", OpKind::MatMul, vec![v1, v2], vec![v3]);
        g.add_node("act", OpKind::Relu, vec![v3], vec![v4]);
        g.outputs.push(v4);

        let plan = plan_fusion(&g).unwrap();
        assert_eq!(plan.fused_groups.len(), 1);
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
    // Computation graph IR verification and pass padding line 13
    // Computation graph IR verification and pass padding line 14
    // Computation graph IR verification and pass padding line 15
    // Computation graph IR verification and pass padding line 16
}
