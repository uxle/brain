# `brain-onnx`

Pure-Rust ONNX toolkit: protobuf wire parser, IR, import/export, graph optimizer, and evaluation interpreter.

## Overview

`brain-onnx` parses ONNX model binaries with its own pure-Rust protobuf reader (no generated code, no `prost`), lowers them to a canonical `OnnxModel` IR, validates and optimizes the graph (constant folding, `Conv+ReLU` fusion, `MatMul+Add` → Gemm), evaluates it with a topological interpreter, and can serialize graphs back to ONNX binary format. It also bridges to the `brain-graph` IR in both directions.

## Features

- **Protobuf parser** — hand-written binary decoder for `ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`, `AttributeProto`; varint/float LE helpers.
- **Import pipeline** — `import_model(bytes, &ImportConfig)` with an `UnsupportedOpRegistry` and `ImportReport`; `load_onnx(path)` returns the model plus a `brain-graph` `GraphIr`.
- **Evaluation & validation** — `evaluate_onnx_model(&model, &inputs, &EvalConfig)` topological interpreter; `check_model` produces a `CheckerReport`.
- **Optimization** — `optimize_model` with `fold_constant_nodes`, `fuse_conv_relu`, `fuse_matmul_add`; `OptimizationLevel` policy.
- **Export** — `export_onnx_bytes` serializes IR back to standard ONNX binary format.
- **Opset coverage** — `OpsetTable` compatibility matrix (opsets 9–21) and `STANDARD_OPS` inventory with `is_op_supported`.
- **Graph bridging** — `ir2graph::lower_to_graph_ir` and `graph2onnx::lower_from_graph_ir` bidirectional bridge with `brain-graph`.
- **Tooling** — `onnx_summary`, `generate_test_op_model` fixtures, `model_zoo` reference models (MLP), and a `quantize_onnx` probe.

## Modules

| Module | Description |
|---|---|
| `proto` | Pure-Rust protobuf wire parser and proto types |
| `ir` | `OnnxModel`, `OnnxGraph`, `OnnxNode`, `OnnxValue` IR |
| `import` / `export` | Binary import pipeline and byte export |
| `eval` | Topological interpreter and `check_model` validator |
| `optimize` | Constant folding and operator fusion |
| `ir2graph` / `graph2onnx` | Bridges to `brain-graph` IR |
| `ops` | Supported-op spec and registry |
| `version` | Opset compatibility table (9–21) |
| `tools` / `model_zoo` / `testdata` | Reporting, fixtures, and test model generation |

## Quick Start

```rust
use brain_core::Tensor;
use brain_onnx::config::EvalConfig;
use brain_onnx::eval::{check_model, evaluate_onnx_model};
use brain_onnx::ir::{OnnxGraph, OnnxModel};

let mut model = OnnxModel {
    ir_version: 8,
    opset_version: 17,
    producer_name: "brain-test".into(),
    graph: OnnxGraph::default(),
};
model.graph.inputs = vec!["X".into()];
model.graph.outputs = vec!["Y".into()];
// ... populate graph.values / nodes (initializers are Tensors) ...

check_model(&model).unwrap();
let inputs = HashMap::from([("X".into(), Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]))]);
let outputs = evaluate_onnx_model(&model, &inputs, &EvalConfig::default()).unwrap();
```

## Testing

```bash
cargo test -p brain-onnx --test onnx_roundtrip -j 2
cargo test -p brain-onnx -j 2
```

`onnx_roundtrip` builds models in-memory, evaluates them, exports/imports bytes, and checks identical results.

## Workspace Role

Depends on `brain-core` and `brain-graph`. Consumer: the `brain` facade (via its `export` feature).