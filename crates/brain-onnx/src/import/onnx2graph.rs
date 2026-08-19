//! # ONNX Proto to IR Lowering
//!
//! Transforms `ModelProto` structures into verified `OnnxModel` graphs with tensor shapes and initializers.
#![allow(missing_docs)]

use crate::core::OnnxResult;
use crate::config::ImportConfig;
use crate::ir::{OnnxModel, OnnxGraph, OnnxValue};
use crate::proto::ModelProto;
use super::ops::translate_op;

/// Converts a decoded ModelProto into an OnnxModel IR.
pub fn proto_to_ir(proto: &ModelProto, _config: &ImportConfig) -> OnnxResult<OnnxModel> {
    let mut model = OnnxModel {
        ir_version: proto.ir_version,
        opset_version: proto.opset_import.first().map(|x| x.1).unwrap_or(17),
        producer_name: proto.producer_name.clone(),
        graph: OnnxGraph::default(),
    };

    if let Some(ref g) = proto.graph {
        model.graph.name = g.name.clone();

        for n in &g.node {
            model.graph.nodes.push(translate_op(n));
        }

        for inp in &g.input {
            model.graph.inputs.push(inp.name.clone());
            model.graph.values.insert(inp.name.clone(), OnnxValue {
                name: inp.name.clone(),
                shape: inp.shape.clone(),
                is_initializer: false,
                tensor_data: None,
            });
        }

        for out in &g.output {
            model.graph.outputs.push(out.name.clone());
            model.graph.values.insert(out.name.clone(), OnnxValue {
                name: out.name.clone(),
                shape: out.shape.clone(),
                is_initializer: false,
                tensor_data: None,
            });
        }

        for init in &g.initializer {
            let tensor = init.to_tensor().ok();
            model.graph.values.insert(init.name.clone(), OnnxValue {
                name: init.name.clone(),
                shape: init.dims.clone(),
                is_initializer: true,
                tensor_data: tensor,
            });
        }
    }

    Ok(model)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
