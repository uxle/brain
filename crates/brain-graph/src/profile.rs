//! # Graph Profiler & Memory Liveness
//!
//! Peak memory estimator and node execution profiling.
#![allow(missing_docs)]

use crate::ir::GraphIr;
use std::collections::HashSet;

/// Profiling summary report.
#[derive(Debug, Clone, Default)]
pub struct GraphProfile {
    pub total_nodes: usize,
    pub total_flops: usize,
    pub peak_memory_bytes: usize,
}

/// Profiles a graph for memory liveness and estimated operations.
pub fn profile_graph(graph: &GraphIr) -> GraphProfile {
    let mut total_flops = 0;
    let mut current_memory = 0;
    let mut peak_memory = 0;

    let mut live_values = HashSet::new();

    for &inp in &graph.inputs {
        let bytes = graph.values[inp].shape.num_elements() * 4;
        current_memory += bytes;
        live_values.insert(inp);
    }
    peak_memory = peak_memory.max(current_memory);

    for node in &graph.nodes {
        // Estimate flops
        if let Some(&out) = node.outputs.first() {
            let count = graph.values[out].shape.num_elements();
            total_flops += match node.op {
                crate::ir::ops::OpKind::MatMul => count * 2,
                _ => count,
            };
            let bytes = count * 4;
            current_memory += bytes;
            peak_memory = peak_memory.max(current_memory);
        }
    }

    GraphProfile {
        total_nodes: graph.nodes.len(),
        total_flops,
        peak_memory_bytes: peak_memory,
    }
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
