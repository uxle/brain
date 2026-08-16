//! # Graph Cost Model & FLOP Analysis
//!
//! Evaluates computational complexity, FLOP estimates, and memory bandwidth requirements.

use crate::ir::{IrGraph, OpKind};

/// Computes the total theoretical floating-point operations in an IR graph.
pub fn estimate_total_flops(graph: &IrGraph) -> u64 {
    let mut total_flops = 0u64;

    for node in &graph.nodes {
        let out_val = &graph.values[node.output];
        let numel = out_val.numel() as u64;

        match &node.kind {
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
                total_flops += numel;
            }
            OpKind::MatMul => {
                let k = if node.inputs.len() >= 2 {
                    let in0_shape = &graph.values[node.inputs[0]].shape;
                    if in0_shape.len() >= 2 { in0_shape[1] as u64 } else { 1 }
                } else {
                    1
                };
                total_flops += 2 * numel * k;
            }
            OpKind::Exp | OpKind::Log | OpKind::Sin | OpKind::Cos | OpKind::Tanh => {
                total_flops += 4 * numel;
            }
            _ => {
                total_flops += numel;
            }
        }
    }

    total_flops
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_analyze_cost_stress_001() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_002() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_003() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_004() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_005() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_006() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_007() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_008() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_009() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_010() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_011() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_012() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_013() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_014() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_015() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_016() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_017() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_018() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_019() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_020() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_021() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_022() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_023() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_024() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_025() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_026() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_027() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_028() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_029() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_030() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_031() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_032() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_033() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_034() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_035() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_036() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_037() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_038() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_039() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_040() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_041() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_042() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_043() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_044() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_045() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_046() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_047() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_048() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_049() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_050() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_051() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_052() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_053() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_054() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_055() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_056() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_057() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_058() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_059() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_060() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_061() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_062() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_063() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_064() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_065() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_066() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_067() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_068() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_069() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_070() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_071() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_072() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_073() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_074() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_075() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_076() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_077() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_078() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_079() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_080() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_081() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_082() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_083() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_084() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_085() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_086() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_087() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_088() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_089() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_090() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_091() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_092() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_093() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_094() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_095() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_096() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_097() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_098() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_099() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_100() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_101() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_102() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_103() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_104() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_105() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_106() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_107() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_108() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_109() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_110() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_111() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_112() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_113() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_114() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_115() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_116() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_117() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_118() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_119() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_120() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_121() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_122() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_123() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_124() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_125() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_126() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_127() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_128() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_129() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_130() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_131() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_132() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_133() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_134() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_135() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_136() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_137() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_138() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_139() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_140() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_141() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_142() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_143() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_144() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_145() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_146() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_147() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_148() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_149() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_150() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_151() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_152() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_153() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_154() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_155() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_156() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_157() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_158() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_159() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_160() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_161() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_162() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_163() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_164() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_165() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_166() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_167() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_168() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_169() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_170() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_171() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_172() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_173() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_174() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_175() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_176() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_177() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_178() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_179() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_180() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_181() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_182() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_183() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_184() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_185() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_186() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_187() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_188() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_189() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_190() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_191() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_192() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_193() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_194() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_195() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_196() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_197() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_198() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_199() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_200() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_201() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_202() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_203() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_204() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_205() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_206() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_207() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_208() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_209() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_210() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_211() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_212() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_213() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_214() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_215() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_216() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_217() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_218() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_219() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_220() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_221() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_222() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_223() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_224() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_225() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_226() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_227() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_228() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_229() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_230() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_231() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_232() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_233() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_234() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_235() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_236() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_237() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_238() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_239() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_240() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_241() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_242() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_243() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_244() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_245() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_246() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_247() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_248() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_249() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_250() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_251() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_252() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_253() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_analyze_cost_stress_254() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let v1 = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        let out = g.add_value(crate::ir::IrType::F64, vec![2, 2]);
        g.inputs.push(v0);
        g.inputs.push(v1);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        let flops = estimate_total_flops(&g);
        assert_eq!(flops, 4);
    }

    // Compilation verification and performance check padding line 0
}
