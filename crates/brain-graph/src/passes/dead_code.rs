//! # Dead Code Elimination (DCE)
//!
//! Removes nodes and values that do not contribute to any graph outputs.
#![allow(missing_docs)]

use std::collections::HashSet;
use crate::core::GraphResult;
use crate::ir::GraphIr;
use super::GraphPass;

/// Dead Code Elimination pass.
#[derive(Debug, Default)]
pub struct DeadCodeElimPass;

impl GraphPass for DeadCodeElimPass {
    fn name(&self) -> &'static str { "DeadCodeElimination" }

    fn run(&mut self, graph: &mut GraphIr) -> GraphResult<bool> {
        eliminate_dead_code(graph)
    }
}

/// Identifies and removes unused nodes from `GraphIr`.
pub fn eliminate_dead_code(graph: &mut GraphIr) -> GraphResult<bool> {
    let mut needed_values: HashSet<usize> = graph.outputs.iter().copied().collect();
    let mut live_nodes = HashSet::new();

    // Traverse in reverse to mark required nodes
    for (node_idx, node) in graph.nodes.iter().enumerate().rev() {
        let produces_needed = node.outputs.iter().any(|out| needed_values.contains(out));
        if produces_needed {
            live_nodes.insert(node_idx);
            for &inp in &node.inputs {
                needed_values.insert(inp);
            }
        }
    }

    let initial_count = graph.nodes.len();
    let mut new_nodes = Vec::new();
    for (idx, node) in graph.nodes.drain(..).enumerate() {
        if live_nodes.contains(&idx) {
            new_nodes.push(node);
        }
    }
    graph.nodes = new_nodes;

    Ok(graph.nodes.len() < initial_count)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
