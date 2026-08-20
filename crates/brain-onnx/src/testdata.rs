//! # Synthetic ONNX Test Data Generators
//!
//! Generates valid ONNX byte streams and graph structures for automated unit and fuzz testing.
#![allow(missing_docs)]

use crate::ir::{OnnxGraph, OnnxModel, OnnxNode};
use std::collections::HashMap;

/// Generates a test OnnxModel with a single operator.
pub fn generate_test_op_model(op_type: &str) -> OnnxModel {
    let mut model = OnnxModel {
        ir_version: 8,
        opset_version: 17,
        producer_name: "brain-test-gen".into(),
        graph: OnnxGraph::default(),
    };

    model.graph.name = format!("test_{}", op_type);
    model.graph.inputs = vec!["in".into()];
    model.graph.outputs = vec!["out".into()];

    model.graph.nodes.push(OnnxNode {
        name: format!("{}_0", op_type),
        op_type: op_type.into(),
        domain: "ai.onnx".into(),
        inputs: vec!["in".into()],
        outputs: vec!["out".into()],
        attributes: HashMap::new(),
    });

    model
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
