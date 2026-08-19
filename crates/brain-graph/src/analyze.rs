//! # Graph Analysis Suite
//!
//! Cycle detection, parallelism factor analysis, and fusion opportunity mining.
#![allow(missing_docs)]

use crate::ir::GraphIr;
use crate::topology::compute_topological_order;
use crate::schedule::generate_schedule;

/// Checks whether the computation graph contains any cycles.
pub fn analyze_cycles(graph: &GraphIr) -> bool {
    compute_topological_order(graph).is_err()
}

/// Measures the average parallelism factor across scheduled execution stages.
pub fn analyze_parallelism(graph: &GraphIr) -> f64 {
    let schedule = generate_schedule(graph);
    if schedule.num_stages() == 0 { return 0.0; }
    let total_nodes: usize = schedule.stages.iter().map(|s| s.len()).sum();
    total_nodes as f64 / schedule.num_stages() as f64
}

/// Finds the count of candidate fusion pairs in the graph.
pub fn analyze_fusion_candidates(graph: &GraphIr) -> usize {
    crate::passes::fusion::plan_fusion(graph).map(|p| p.fused_groups.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
