//! # Execution Scheduling
//!
//! Level-based batching and parallel region partition for multicore / JIT execution.
#![allow(missing_docs)]

use crate::core::NodeId;
use crate::ir::GraphIr;
use crate::topology::compute_topological_order;
use std::collections::HashMap;

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

    let stages = sorted_keys
        .into_iter()
        .map(|k| stages_map.remove(&k).unwrap())
        .collect();

    SchedulePlan { stages }
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
