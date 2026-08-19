//! # Exporting Brain Graph IR to ONNX Model
//!
//! Reverse lowering translating `brain_graph::GraphIr` structures back into `OnnxModel` IR.
#![allow(missing_docs)]

use crate::core::OnnxResult;
use crate::ir::{OnnxModel, OnnxGraph, OnnxNode};
use brain_graph::GraphIr;
use std::collections::HashMap;

/// Translates Brain GraphIr into canonical OnnxModel IR.
pub fn lower_from_graph_ir(graph_ir: &GraphIr) -> OnnxResult<OnnxModel> {
    let mut model = OnnxModel {
        ir_version: 8,
        opset_version: 17,
        producer_name: "brain-onnx-export".into(),
        graph: OnnxGraph::default(),
    };

    model.graph.name = graph_ir.name.clone();

    for node in &graph_ir.nodes {
        model.graph.nodes.push(OnnxNode {
            name: node.name.clone(),
            op_type: format!("{:?}", node.op),
            domain: "ai.onnx".into(),
            inputs: Vec::new(),
            outputs: vec![format!("{}_out", node.name)],
            attributes: HashMap::new(),
        });
    }

    Ok(model)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
