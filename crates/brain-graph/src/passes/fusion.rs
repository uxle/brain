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
}
