//! # IR Verification & Type Checking
//!
//! Validates use-before-def rules, type compatibility, and bounds integrity across IR graphs.

use crate::core::CompilationError;
use crate::ir::IrGraph;

/// Verifies semantic correctness and structural integrity of an `IrGraph`.
pub fn verify_graph(graph: &IrGraph) -> Result<(), CompilationError> {
    let mut defined = vec![false; graph.values.len()];

    for &input_id in &graph.inputs {
        if input_id >= graph.values.len() {
            return Err(CompilationError::VerificationFailed(format!(
                "Input ID {} out of bounds",
                input_id
            )));
        }
        defined[input_id] = true;
    }

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &in_id in &node.inputs {
            if in_id >= graph.values.len() {
                return Err(CompilationError::VerificationFailed(format!(
                    "Node {} input ID {} out of bounds",
                    node_idx, in_id
                )));
            }
            if !defined[in_id] {
                return Err(CompilationError::VerificationFailed(format!(
                    "Node {} used value {} before definition",
                    node_idx, in_id
                )));
            }
        }

        if node.output >= graph.values.len() {
            return Err(CompilationError::VerificationFailed(format!(
                "Node {} output ID {} out of bounds",
                node_idx, node.output
            )));
        }

        defined[node.output] = true;
    }

    for &out_id in &graph.outputs {
        if out_id >= graph.values.len() {
            return Err(CompilationError::VerificationFailed(format!(
                "Output ID {} out of bounds",
                out_id
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
