//! # ONNX Graph Model Export & Evaluation Example
//!
//! Demonstrates constructing, checking, and evaluating an ONNX graph in pure Rust.

use brain_core::Tensor;
use brain_onnx::config::EvalConfig;
use brain_onnx::eval::{check_model, evaluate_onnx_model};
use brain_onnx::ir::{OnnxGraph, OnnxModel, OnnxNode, OnnxValue};
use std::collections::HashMap;

fn main() {
    println!("=== Brain 1.0 ONNX Model Export & Eval Example ===");

    let mut model = OnnxModel {
        ir_version: 8,
        opset_version: 17,
        producer_name: "brain-example".into(),
        graph: OnnxGraph::default(),
    };

    model.graph.name = "linear_layer".into();
    model.graph.inputs = vec!["X".into()];
    model.graph.outputs = vec!["Y".into()];

    model.graph.values.insert("X".into(), OnnxValue {
        name: "X".into(),
        shape: vec![1, 2],
        is_initializer: false,
        tensor_data: None,
    });
    model.graph.values.insert("W".into(), OnnxValue {
        name: "W".into(),
        shape: vec![2, 2],
        is_initializer: true,
        tensor_data: Some(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])),
    });
    model.graph.values.insert("Y".into(), OnnxValue {
        name: "Y".into(),
        shape: vec![1, 2],
        is_initializer: false,
        tensor_data: None,
    });

    model.graph.nodes.push(OnnxNode {
        name: "matmul_op".into(),
        op_type: "MatMul".into(),
        domain: "ai.onnx".into(),
        inputs: vec!["X".into(), "W".into()],
        outputs: vec!["Y".into()],
        attributes: HashMap::new(),
    });

    let report = check_model(&model).expect("Graph validation");
    assert!(report.is_valid, "Valid model");
    println!("ONNX model validated successfully (opset 17).");

    let mut inputs = HashMap::new();
    inputs.insert("X".into(), Tensor::from_vec(vec![1.0, 1.0], vec![1, 2]));

    let outputs = evaluate_onnx_model(&model, &inputs, &EvalConfig::default()).expect("Eval");
    let y = outputs.get("Y").expect("Output Y");
    println!("Evaluated output Y: {:?}", y.to_vec());
}
