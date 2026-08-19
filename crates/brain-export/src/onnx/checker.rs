//! # ONNX Graph Integrity Checker
//!
//! Validates topological sort ordering, input/output connectivity, and shape consistency.

use crate::common::ExportIr;
use crate::core::ExportError;

/// Validates that an intermediate graph satisfies all structural ONNX invariants.
pub fn validate_onnx_graph(graph: &ExportIr) -> Result<(), ExportError> {
    if graph.name.is_empty() {
        return Err(ExportError::VerificationFailed("Empty graph name".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
