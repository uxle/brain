//! # Tensor Lifetime & Computation Intensity Analysis
//!
//! Tracks live ranges of tensor registers and estimates arithmetic intensity.

use crate::ir::IrGraph;

/// Analyzes tensor lifetime spans in an IR graph.
pub fn analyze_tensor_lifetimes(graph: &IrGraph) -> Vec<(usize, usize)> {
    let mut lifetimes = vec![(0, 0); graph.values.len()];
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        lifetimes[node.output].0 = node_idx;
        lifetimes[node.output].1 = node_idx;
        for &inp in &node.inputs {
            lifetimes[inp].1 = node_idx;
        }
    }
    lifetimes
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
