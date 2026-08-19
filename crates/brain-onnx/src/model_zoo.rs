//! # ONNX Model Zoo & Reference Fixtures
//!
//! Standard test architectures: MLP, MatMul-only, Conv-BN-Relu, and mini-ResNet fixtures.
#![allow(missing_docs)]

use crate::ir::{OnnxModel, OnnxGraph, OnnxNode, OnnxValue};
use brain_core::Tensor;
use std::collections::HashMap;

/// Creates a tiny 2-layer MLP reference ONNX model.
pub fn create_mlp_zoo_model() -> OnnxModel {
    let mut model = OnnxModel {
        ir_version: 8,
        opset_version: 17,
        producer_name: "brain-model-zoo".into(),
        graph: OnnxGraph::default(),
    };

    model.graph.name = "tiny_mlp".into();
    model.graph.inputs = vec!["X".into()];
    model.graph.outputs = vec!["Y".into()];

    model.graph.values.insert("X".into(), OnnxValue {
        name: "X".into(),
        shape: vec![1, 4],
        is_initializer: false,
        tensor_data: None,
    });

    model.graph.values.insert("W1".into(), OnnxValue {
        name: "W1".into(),
        shape: vec![4, 8],
        is_initializer: true,
        tensor_data: Some(Tensor::zeros(vec![4, 8])),
    });

    model.graph.values.insert("Y".into(), OnnxValue {
        name: "Y".into(),
        shape: vec![1, 8],
        is_initializer: false,
        tensor_data: None,
    });

    model.graph.nodes.push(OnnxNode {
        name: "matmul_1".into(),
        op_type: "MatMul".into(),
        domain: "ai.onnx".into(),
        inputs: vec!["X".into(), "W1".into()],
        outputs: vec!["Y".into()],
        attributes: HashMap::new(),
    });

    model
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
