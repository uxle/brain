//! # ONNX Model Inspection & Diagnostic Tools
//!
//! Model summary generation, operator inventory counting, tensor size stats, and structural reports.
#![allow(missing_docs)]

use crate::ir::OnnxModel;
use std::collections::HashMap;

/// Generates a textual summary of an ONNX model.
pub fn onnx_summary(model: &OnnxModel) -> String {
    let total_nodes = model.graph.nodes.len();
    let total_inputs = model.graph.inputs.len();
    let total_outputs = model.graph.outputs.len();

    let mut op_counts: HashMap<String, usize> = HashMap::new();
    for node in &model.graph.nodes {
        *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
    }

    format!(
        "ONNX Model Summary:
  IR Version: {}
  Opset: {}
  Nodes: {}
  Inputs: {}
  Outputs: {}
  Operator Breakdown: {:?}",
        model.ir_version, model.opset_version, total_nodes, total_inputs, total_outputs, op_counts
    )
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
