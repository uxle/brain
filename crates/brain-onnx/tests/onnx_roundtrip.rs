//! # ONNX Model Roundtrip & Evaluation Verification Tests

use brain_core::Tensor;
use brain_onnx::config::EvalConfig;
use brain_onnx::eval::{check_model, evaluate_onnx_model};
use brain_onnx::ir::{OnnxGraph, OnnxModel, OnnxNode, OnnxValue};
use std::collections::HashMap;

#[test]
fn test_onnx_mlp_roundtrip_eval() {
    let mut model = OnnxModel {
        ir_version: 8,
        opset_version: 17,
        producer_name: "brain-test".into(),
        graph: OnnxGraph::default(),
    };

    model.graph.name = "linear_relu".into();
    model.graph.inputs = vec!["X".into()];
    model.graph.outputs = vec!["Y".into()];

    let w_data = vec![0.5, -0.5, 1.0, 2.0]; // [2, 2]
    let b_data = vec![0.1, -0.1]; // [2]

    model.graph.values.insert(
        "X".into(),
        OnnxValue {
            name: "X".into(),
            shape: vec![1, 2],
            is_initializer: false,
            tensor_data: None,
        },
    );
    model.graph.values.insert(
        "W".into(),
        OnnxValue {
            name: "W".into(),
            shape: vec![2, 2],
            is_initializer: true,
            tensor_data: Some(Tensor::from_vec(w_data.clone(), vec![2, 2])),
        },
    );
    model.graph.values.insert(
        "B".into(),
        OnnxValue {
            name: "B".into(),
            shape: vec![1, 2],
            is_initializer: true,
            tensor_data: Some(Tensor::from_vec(b_data.clone(), vec![1, 2])),
        },
    );
    model.graph.values.insert(
        "MM".into(),
        OnnxValue {
            name: "MM".into(),
            shape: vec![1, 2],
            is_initializer: false,
            tensor_data: None,
        },
    );
    model.graph.values.insert(
        "ADD".into(),
        OnnxValue {
            name: "ADD".into(),
            shape: vec![1, 2],
            is_initializer: false,
            tensor_data: None,
        },
    );
    model.graph.values.insert(
        "Y".into(),
        OnnxValue {
            name: "Y".into(),
            shape: vec![1, 2],
            is_initializer: false,
            tensor_data: None,
        },
    );

    model.graph.nodes.push(OnnxNode {
        name: "mm_1".into(),
        op_type: "MatMul".into(),
        domain: "ai.onnx".into(),
        inputs: vec!["X".into(), "W".into()],
        outputs: vec!["MM".into()],
        attributes: HashMap::new(),
    });
    model.graph.nodes.push(OnnxNode {
        name: "add_1".into(),
        op_type: "Add".into(),
        domain: "ai.onnx".into(),
        inputs: vec!["MM".into(), "B".into()],
        outputs: vec!["ADD".into()],
        attributes: HashMap::new(),
    });
    model.graph.nodes.push(OnnxNode {
        name: "relu_1".into(),
        op_type: "Relu".into(),
        domain: "ai.onnx".into(),
        inputs: vec!["ADD".into()],
        outputs: vec!["Y".into()],
        attributes: HashMap::new(),
    });

    let report = check_model(&model).unwrap();
    assert!(report.is_valid, "Model check failed: {:?}", report.errors);

    let x = Tensor::from_vec(vec![2.0, 1.0], vec![1, 2]);
    let mut inputs = HashMap::new();
    inputs.insert("X".into(), x.clone());

    let outputs = evaluate_onnx_model(&model, &inputs, &EvalConfig::default()).unwrap();
    let y = outputs.get("Y").unwrap();

    // Reference Brain forward:
    // MM = [2.0, 1.0] @ [[0.5, -0.5], [1.0, 2.0]] = [2.0*0.5 + 1.0*1.0, 2.0*(-0.5) + 1.0*2.0] = [2.0, 1.0]
    // ADD = [2.0 + 0.1, 1.0 - 0.1] = [2.1, 0.9]
    // Relu = [2.1, 0.9]
    assert_eq!(y.shape(), &[1, 2]);
    let y_vec = y.to_vec();
    assert!(
        (y_vec[0] - 2.1).abs() < 1e-5,
        "Expected 2.1, got {}",
        y_vec[0]
    );
    assert!(
        (y_vec[1] - 0.9).abs() < 1e-5,
        "Expected 0.9, got {}",
        y_vec[1]
    );

    // Test real export -> import binary round-trip:
    let bytes = brain_onnx::export::export_onnx_bytes(&model).unwrap();
    assert!(!bytes.is_empty());
    let imported =
        brain_onnx::import::import_model(&bytes, &brain_onnx::config::ImportConfig::default())
            .unwrap();
    assert_eq!(imported.graph.nodes.len(), 3);
    assert_eq!(imported.graph.inputs, vec!["X".to_string()]);
    assert_eq!(imported.graph.outputs, vec!["Y".to_string()]);

    let outputs_imported = evaluate_onnx_model(&imported, &inputs, &EvalConfig::default()).unwrap();
    let y_imp = outputs_imported.get("Y").unwrap();
    let y_imp_vec = y_imp.to_vec();
    assert!((y_imp_vec[0] - 2.1).abs() < 1e-5);
    assert!((y_imp_vec[1] - 0.9).abs() < 1e-5);
}

#[test]
fn test_onnx_eval_matmul_non_identity() {
    let mut model = OnnxModel {
        ir_version: 8,
        opset_version: 17,
        producer_name: "brain-test".into(),
        graph: OnnxGraph::default(),
    };
    model.graph.name = "matmul_test".into();
    model.graph.inputs = vec!["X".into()];
    model.graph.outputs = vec!["Y".into()];

    let w_data = vec![3.0, 4.0, 5.0, 6.0]; // [2, 2]
    model.graph.values.insert(
        "W".into(),
        OnnxValue {
            name: "W".into(),
            shape: vec![2, 2],
            is_initializer: true,
            tensor_data: Some(Tensor::from_vec(w_data, vec![2, 2])),
        },
    );

    model.graph.nodes.push(OnnxNode {
        name: "mm".into(),
        op_type: "MatMul".into(),
        domain: "ai.onnx".into(),
        inputs: vec!["X".into(), "W".into()],
        outputs: vec!["Y".into()],
        attributes: HashMap::new(),
    });

    // X = [[1.0, 2.0]]
    // X @ W = [[1*3 + 2*5, 1*4 + 2*6]] = [[13.0, 16.0]]
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
    let mut inputs = HashMap::new();
    inputs.insert("X".into(), x);

    let outputs = evaluate_onnx_model(&model, &inputs, &EvalConfig::default()).unwrap();
    let y = outputs.get("Y").unwrap();
    assert_eq!(y.to_vec(), vec![13.0, 16.0]);
}
