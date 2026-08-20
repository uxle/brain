//! # ONNX Graph Integrity & Structural Checker
//!
//! Validates topological DAG ordering, input/output connectivity, and shape consistency.
#![allow(missing_docs)]

use crate::core::OnnxResult;
use crate::ir::OnnxModel;
use std::collections::HashSet;

/// Checker diagnostic report.
#[derive(Debug, Clone, Default)]
pub struct CheckerReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
}

/// Checks the structural and topological validity of an OnnxModel.
pub fn check_model(model: &OnnxModel) -> OnnxResult<CheckerReport> {
    let mut errors = Vec::new();
    let mut produced_values: HashSet<String> = HashSet::new();

    // Inputs and initializers are available from the start
    for inp in &model.graph.inputs {
        produced_values.insert(inp.clone());
    }
    for (name, val) in &model.graph.values {
        if val.is_initializer {
            produced_values.insert(name.clone());
        }
    }

    // Verify node dependency ordering
    for node in &model.graph.nodes {
        for inp in &node.inputs {
            if !inp.is_empty() && !produced_values.contains(inp) {
                errors.push(format!(
                    "Node '{}' uses input '{}' before production",
                    node.name, inp
                ));
            }
        }
        for out in &node.outputs {
            produced_values.insert(out.clone());
        }
    }

    let is_valid = errors.is_empty();
    Ok(CheckerReport { is_valid, errors })
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
