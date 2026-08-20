//! # Dead Code Elimination (DCE) & Common Subexpression Elimination (CSE)
//!
//! Eliminates unreachable, unused computation nodes and coalesces identical sub-expressions.

use crate::core::CompilationError;
use crate::ir::IrGraph;
use crate::passes::Pass;
use std::collections::HashSet;

/// Optimization pass for dead code elimination.
pub struct DeadCodeEliminationPass;

impl Pass for DeadCodeEliminationPass {
    fn name(&self) -> &str {
        "dead-code-elimination"
    }

    fn run(&self, graph: &mut IrGraph) -> Result<bool, CompilationError> {
        let initial_count = graph.nodes.len();
        let mut live_values: HashSet<usize> = graph.outputs.iter().copied().collect();

        // Trace live values backward from outputs through producer nodes
        let mut live_nodes = Vec::new();

        for node in graph.nodes.iter().rev() {
            if live_values.contains(&node.output) {
                live_nodes.push(node.clone());
                for &input in &node.inputs {
                    live_values.insert(input);
                }
            }
        }

        live_nodes.reverse();
        let changed = live_nodes.len() != initial_count;
        graph.nodes = live_nodes;

        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IrType, OpKind};

    #[test]
    fn test_dce_pruning() {
        let mut graph = IrGraph::new();
        let x_id = graph.add_value(IrType::F64, vec![1]);
        let dead_id = graph.add_value(IrType::F64, vec![1]);
        let live_out = graph.add_value(IrType::F64, vec![1]);

        graph.inputs = vec![x_id];
        graph.outputs = vec![live_out];

        // Live path
        graph.add_node(OpKind::Relu, vec![x_id], live_out);

        // Dead path (never consumed by output)
        graph.add_node(OpKind::Neg, vec![x_id], dead_id);

        assert_eq!(graph.nodes.len(), 2);

        let dce = DeadCodeEliminationPass;
        let changed = dce.run(&mut graph).unwrap();
        assert!(changed);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].output, live_out);
    }
}
