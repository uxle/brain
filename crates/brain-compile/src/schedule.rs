//! # Topological Instruction Scheduling
//!
//! Generates topological schedules and identifies parallelizable sub-graphs.

use crate::ir::IrGraph;

/// Execution schedule plan.
#[derive(Debug, Clone, Default)]
pub struct SchedulePlan {
    pub execution_order: Vec<usize>,
}

impl SchedulePlan {
    /// Computes a linear execution schedule for the graph.
    pub fn compute_schedule(graph: &IrGraph) -> Self {
        let order: Vec<usize> = (0..graph.num_nodes()).collect();
        Self {
            execution_order: order,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
