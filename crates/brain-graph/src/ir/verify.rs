//! # Graph IR Verification
//!
//! Topological sanity checks, defined-before-use verification, and cycle detection.
#![allow(missing_docs)]

use std::collections::HashSet;
use crate::core::{GraphError, GraphResult};
use crate::ir::GraphIr;

/// Verifies structural and semantic integrity of a `GraphIr`.
pub fn verify_graph(graph: &GraphIr) -> GraphResult<()> {
    let mut defined_values = HashSet::new();

    // Graph inputs are pre-defined
    for &input in &graph.inputs {
        if input >= graph.values.len() {
            return Err(GraphError::ValueNotFound(input));
        }
        defined_values.insert(input);
    }

    // Constants are also defined
    for (i, v) in graph.values.iter().enumerate() {
        if v.constant_data.is_some() {
            defined_values.insert(i);
        }
    }

    // Validate nodes in topological sequence
    for node in &graph.nodes {
        // Check minimum inputs
        if node.inputs.len() < node.op.min_inputs() {
            return Err(GraphError::VerificationFailed(format!(
                "Node '{}' (op {:?}) has {} inputs, expected at least {}",
                node.name, node.op, node.inputs.len(), node.op.min_inputs()
            )));
        }

        // Check input values defined before use
        for &inp in &node.inputs {
            if !defined_values.contains(&inp) {
                return Err(GraphError::VerificationFailed(format!(
                    "Value {} used in node '{}' before definition",
                    inp, node.name
                )));
            }
        }

        // Register outputs as defined
        for &out in &node.outputs {
            if out >= graph.values.len() {
                return Err(GraphError::ValueNotFound(out));
            }
            defined_values.insert(out);
        }
    }

    // Validate graph outputs
    for &output in &graph.outputs {
        if !defined_values.contains(&output) {
            return Err(GraphError::VerificationFailed(format!(
                "Graph output value {} was never defined", output
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
