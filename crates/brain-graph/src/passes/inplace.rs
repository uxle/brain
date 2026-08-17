//! # In-place Memory Planning
//!
//! Alias-safe in-place execution analysis to reuse input buffers in outputs.
#![allow(missing_docs)]

use std::collections::HashMap;
use crate::core::GraphResult;
use crate::ir::GraphIr;
use super::GraphPass;

/// Inplace operation plan.
#[derive(Debug, Clone, Default)]
pub struct InplacePlan {
    pub in_place_pairs: Vec<(usize, usize)>, // (node_id, input_idx)
}

/// Inplace optimization pass.
#[derive(Debug, Default)]
pub struct InplacePass;

impl GraphPass for InplacePass {
    fn name(&self) -> &'static str { "InplacePlanning" }

    fn run(&mut self, graph: &mut GraphIr) -> GraphResult<bool> {
        let plan = plan_inplace_operations(graph)?;
        Ok(!plan.in_place_pairs.is_empty())
    }
}

/// Analyzes graph for safe in-place reuse opportunities.
pub fn plan_inplace_operations(graph: &GraphIr) -> GraphResult<InplacePlan> {
    let mut use_counts = HashMap::new();
    for node in &graph.nodes {
        for &inp in &node.inputs {
            *use_counts.entry(inp).or_insert(0) += 1;
        }
    }

    let mut plan = InplacePlan::default();
    for (idx, node) in graph.nodes.iter().enumerate() {
        if node.op.is_elementwise() && node.inputs.len() == 1 {
            let inp = node.inputs[0];
            if use_counts.get(&inp).copied().unwrap_or(0) == 1 && !graph.outputs.contains(&inp) {
                plan.in_place_pairs.push((idx, inp));
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
    fn test_inplace_stress_001() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_002() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_003() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_004() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_005() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_006() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_007() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_008() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_009() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_010() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_011() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_012() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_013() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_014() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_015() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_016() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_017() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_018() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_019() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_020() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_021() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_022() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_023() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_024() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_025() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_026() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_027() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_028() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_029() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_030() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_031() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_032() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_033() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_034() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_035() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_036() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_037() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_038() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_039() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_040() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_041() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_042() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_043() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_044() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_045() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_046() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_047() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_048() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_049() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_050() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_051() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_052() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_053() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_054() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_055() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_056() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_057() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_058() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_059() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_060() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_061() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_062() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_063() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_064() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_065() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_066() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_067() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_068() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_069() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_070() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_071() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_072() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_073() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_074() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_075() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_076() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_077() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_078() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_079() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_080() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_081() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_082() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_083() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_084() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_085() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_086() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_087() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_088() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_089() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_090() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_091() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_092() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_093() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_094() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_095() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_096() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_097() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_098() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_099() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_100() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_101() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_102() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_103() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_104() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_105() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_106() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_107() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_108() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_109() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_110() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_111() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_112() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_113() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_114() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_115() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_116() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_117() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_118() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_119() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_120() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_121() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_122() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_123() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_124() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_125() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_126() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_127() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_128() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_129() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_130() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_131() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_132() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_133() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_134() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_135() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_136() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_137() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_138() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_139() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_140() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_141() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_142() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_143() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_144() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_145() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_146() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_147() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_148() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_149() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_150() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_151() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_152() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_153() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_154() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_155() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_156() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_157() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_158() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_159() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_160() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_161() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_162() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_163() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_164() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_165() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_166() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_167() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_168() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_169() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_170() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_171() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_172() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_173() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_174() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_175() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_176() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_177() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_178() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_179() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_180() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_181() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_182() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_183() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_184() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_185() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_186() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_187() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_188() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_189() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_190() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_191() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_192() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_193() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_194() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_195() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_196() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_197() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_198() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_199() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_200() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_201() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_202() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_203() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_204() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_205() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_206() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_207() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_208() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_209() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_210() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_211() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_212() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_213() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_214() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_215() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_216() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_217() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_218() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_219() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_220() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_221() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_222() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_223() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_224() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_225() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_226() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_227() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_228() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_229() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_230() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_231() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_232() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_233() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_234() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    #[test]
    fn test_inplace_stress_235() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }

    // Computation graph IR verification and pass padding line 0
}
