//! # Graph Diffing & Equivalence
//!
//! Structural and semantic comparison between computation graphs.
#![allow(missing_docs)]

use crate::ir::GraphIr;

/// Difference report between two computation graphs.
#[derive(Debug, Clone, Default)]
pub struct GraphDiff {
    pub added_nodes: usize,
    pub removed_nodes: usize,
    pub modified_nodes: usize,
    pub is_structurally_identical: bool,
}

/// Compares two graphs structurally.
pub fn diff_graphs(a: &GraphIr, b: &GraphIr) -> GraphDiff {
    let mut diff = GraphDiff::default();
    if a.nodes.len() < b.nodes.len() {
        diff.added_nodes = b.nodes.len() - a.nodes.len();
    } else {
        diff.removed_nodes = a.nodes.len() - b.nodes.len();
    }

    let min_len = a.nodes.len().min(b.nodes.len());
    for i in 0..min_len {
        if a.nodes[i].op != b.nodes[i].op {
            diff.modified_nodes += 1;
        }
    }

    diff.is_structurally_identical =
        diff.added_nodes == 0 && diff.removed_nodes == 0 && diff.modified_nodes == 0;
    diff
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
