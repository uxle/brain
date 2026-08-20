//! # In-place Memory Planning
//!
//! Alias-safe in-place execution analysis to reuse input buffers in outputs.
#![allow(missing_docs)]

use super::GraphPass;
use crate::core::GraphResult;
use crate::ir::GraphIr;
use std::collections::HashMap;

/// Inplace operation plan.
#[derive(Debug, Clone, Default)]
pub struct InplacePlan {
    pub in_place_pairs: Vec<(usize, usize)>, // (node_id, input_idx)
}

/// Inplace optimization pass.
#[derive(Debug, Default)]
pub struct InplacePass;

impl GraphPass for InplacePass {
    fn name(&self) -> &'static str {
        "InplacePlanning"
    }

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
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
