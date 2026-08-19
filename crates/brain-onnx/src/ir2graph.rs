//! # Lowering ONNX IR to Brain Graph IR
//!
//! Translates `OnnxModel` into `brain_graph::GraphIr` for compilation, fusion, and execution.
#![allow(missing_docs)]

use crate::core::{OnnxError, OnnxResult};
use crate::ir::OnnxModel;
use brain_graph::GraphIr;
use brain_graph::builder::GraphBuilder;
use brain_graph::core::DType;
use brain_graph::ir::ops::OpKind;
use std::collections::HashMap;

/// Lowers an OnnxModel into Brain GraphIr.
pub fn lower_to_graph_ir(model: &OnnxModel) -> OnnxResult<GraphIr> {
    let name = if model.graph.name.is_empty() { "onnx_model" } else { &model.graph.name };
    if model.graph.nodes.is_empty() && model.graph.inputs.is_empty() {
        return Ok(GraphIr::new(name));
    }
    let mut builder = GraphBuilder::new(name);
    let mut value_map = HashMap::new();

    // Add inputs
    for inp in &model.graph.inputs {
        let shape = model.graph.values.get(inp).map(|v| v.shape.clone()).unwrap_or_else(|| vec![1, 1]);
        let vid = builder.add_input(inp, shape, DType::F32);
        value_map.insert(inp.clone(), vid);
    }

    // Add initializers
    for (val_name, val) in &model.graph.values {
        if val.is_initializer {
            let data = val.tensor_data.as_ref().map(|t| t.to_vec()).unwrap_or_default();
            let shape = if val.shape.is_empty() { vec![1] } else { val.shape.clone() };
            let vid = builder.add_constant(val_name, shape, data);
            value_map.insert(val_name.clone(), vid);
        }
    }

    // Add operators
    for node in &model.graph.nodes {
        let op_kind = match node.op_type.as_str() {
            "Add" => OpKind::Add,
            "Sub" => OpKind::Sub,
            "Mul" => OpKind::Mul,
            "Div" => OpKind::Div,
            "MatMul" => OpKind::MatMul,
            "Relu" => OpKind::Relu,
            "Sigmoid" => OpKind::Sigmoid,
            "Tanh" => OpKind::Tanh,
            _ => OpKind::Custom,
        };

        let node_inputs: Vec<_> = node.inputs.iter().filter_map(|inp| value_map.get(inp).copied()).collect();
        let out_name = node.outputs.first().cloned().unwrap_or_else(|| format!("{}_out", node.name));
        let out_id = builder.add_node(&node.name, op_kind, node_inputs, vec![1, 1]);
        value_map.insert(out_name, out_id);
    }

    // Mark outputs
    for out in &model.graph.outputs {
        if let Some(&vid) = value_map.get(out) {
            builder.mark_output(vid);
        }
    }

    builder.build().map_err(|e| OnnxError::GraphLoweringError(e.to_string()))
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
