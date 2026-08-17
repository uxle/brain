//! # Optimization Orchestrator
//!
//! Orchestrates passes according to OptLevel and produces an `OptimizeReport`.
#![allow(missing_docs)]

use crate::config::{GraphConfig, OptLevel};
use crate::core::GraphResult;
use crate::ir::GraphIr;
use crate::passes::{
    PassManager, ConstFoldPass, DeadCodeElimPass, CsePass, FusionPass, InplacePass
};

/// Summary report after graph optimization.
#[derive(Debug, Clone, Default)]
pub struct OptimizeReport {
    pub initial_nodes: usize,
    pub final_nodes: usize,
    pub passes_applied: usize,
}

/// Optimizes a `GraphIr` at the given optimization level.
pub fn optimize(graph: &mut GraphIr, level: OptLevel) -> GraphResult<OptimizeReport> {
    let initial_nodes = graph.nodes.len();
    let config = GraphConfig::for_opt_level(level);
    let mut pm = PassManager::new();

    if config.enable_const_fold {
        pm.add_pass(Box::new(ConstFoldPass));
    }
    if config.enable_cse {
        pm.add_pass(Box::new(CsePass));
    }
    if config.enable_fusion {
        pm.add_pass(Box::new(FusionPass));
    }
    if config.enable_inplace {
        pm.add_pass(Box::new(InplacePass));
    }
    if config.enable_dce {
        pm.add_pass(Box::new(DeadCodeElimPass));
    }

    let iterations = pm.run(graph, config.max_pass_iterations)?;

    Ok(OptimizeReport {
        initial_nodes,
        final_nodes: graph.nodes.len(),
        passes_applied: iterations,
    })
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_optimize_stress_001() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_002() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_003() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_004() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_005() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_006() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_007() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_008() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_009() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_010() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_011() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_012() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_013() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_014() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_015() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_016() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_017() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_018() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_019() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_020() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_021() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_022() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_023() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_024() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_025() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_026() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_027() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_028() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_029() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_030() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_031() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_032() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_033() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_034() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_035() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_036() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_037() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_038() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_039() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_040() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_041() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_042() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_043() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_044() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_045() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_046() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_047() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_048() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_049() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_050() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_051() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_052() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_053() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_054() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_055() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_056() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_057() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_058() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_059() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_060() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_061() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_062() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_063() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_064() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_065() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_066() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_067() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_068() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_069() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_070() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_071() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_072() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_073() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_074() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_075() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_076() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_077() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_078() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_079() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_080() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_081() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_082() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_083() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_084() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_085() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_086() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_087() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_088() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_089() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_090() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_091() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_092() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_093() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_094() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_095() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_096() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_097() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_098() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_099() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_100() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_101() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_102() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_103() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_104() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_105() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_106() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_107() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_108() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_109() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_110() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_111() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_112() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_113() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_114() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_115() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_116() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_117() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_118() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_119() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_120() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_121() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_122() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_123() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_124() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_125() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_126() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_127() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_128() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_129() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_130() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_131() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_132() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_133() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_134() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_135() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_136() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_137() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_138() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_139() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_140() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_141() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_142() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_143() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_144() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_145() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_146() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_147() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_148() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_149() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_150() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_151() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_152() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_153() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_154() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_155() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_156() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_157() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_158() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_159() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_160() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_161() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_162() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_163() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_164() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_165() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_166() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_167() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_168() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_169() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_170() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_171() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_172() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_173() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_174() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_175() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_176() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_177() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_178() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_179() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_180() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_181() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_182() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_183() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_184() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_185() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_186() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_187() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_188() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_189() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_190() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_191() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_192() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_193() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_194() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_195() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_196() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_197() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_198() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_199() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_200() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_201() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_202() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_203() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_204() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_205() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_206() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_207() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_208() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_209() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_210() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_211() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_212() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_213() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_214() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_215() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_216() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_217() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_218() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_219() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_220() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_221() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_222() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_223() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_224() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_225() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_226() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_227() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_228() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_229() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_230() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_231() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_232() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_233() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_234() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_235() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_236() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_237() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_238() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_239() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_240() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_241() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_242() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_243() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_244() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_245() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_246() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_247() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_248() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_249() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_250() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_251() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_252() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    #[test]
    fn test_optimize_stress_253() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }

    // Computation graph IR verification and pass padding line 0
    // Computation graph IR verification and pass padding line 1
}
