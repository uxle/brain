# `brain-onnx` (v0.2.0)

> Complete ONNX Import Pipeline, Hand-Rolled Protobuf Binary Wire Parser, Graph Optimization, Topological Evaluation Interpreter, and `brain-graph` Lowering Bridge.

## Overview

`brain-onnx` provides end-to-end ONNX model loading, verification, graph rewriting, and evaluation in 100% pure, safe Rust with zero external runtime dependencies. It features a hand-rolled Protocol Buffers binary wire decoder (`ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`), an intermediate representation (`OnnxModel`), optimization passes (`Conv+Relu`, `MatMul+Add` $\to$ Gemm, constant folding), a topological evaluation engine, and a bidirectional lowering bridge to `brain-graph::GraphIr`.

## Architecture

| Module | Description |
|---|---|
| [`proto`](src/proto/mod.rs) | Zero-dependency Protobuf wire decoder for `ModelProto`, `GraphProto`, `NodeProto`, `TensorProto` |
| [`proto/tensor`](src/proto/tensor.rs) | `TensorProto` raw and typed data array decoding into `brain_core::Tensor` |
| [`proto/attrs`](src/proto/attrs.rs) | `AttributeProto` typed extraction (ints, floats, strings, tensors) with default fallback |
| [`ir`](src/ir/mod.rs) | Canonical ONNX IR structures: `OnnxModel`, `OnnxGraph`, `OnnxNode`, `OnnxValue` |
| [`import`](src/import/mod.rs) | `import_model(bytes)` pipeline converting wire bytes to validated `OnnxModel` IR |
| [`import/ops`](src/import/ops.rs) | Operator and attribute translation (`Conv`, `Gemm`, `MatMul`, `Add`, `Relu`, `Reshape`) |
| [`import/onnx2graph`](src/import/onnx2graph.rs) | Lowering `ModelProto` to `OnnxModel` with initializers and shape metadata |
| [`import/unsupported`](src/import/unsupported.rs) | `UnsupportedOpRegistry` tracking non-standard or missing operator diagnostics |
| [`optimize`](src/optimize/mod.rs) | Graph optimization manager: constant folding, `fuse_conv_relu`, `fuse_matmul_add` ($\to$ Gemm) |
| [`eval`](src/eval/mod.rs) | Direct interpretive graph execution engine against `brain-core` tensors |
| [`eval/checker`](src/eval/checker.rs) | `check_model` validating topological DAG ordering and connectivity |
| [`export`](src/export/mod.rs) | `export_onnx_bytes` serializing `OnnxModel` back to binary protobuf format |
| [`ir2graph`](src/ir2graph.rs) | `lower_to_graph_ir` lowering `OnnxModel` into `brain_graph::GraphIr` |
| [`graph2onnx`](src/graph2onnx.rs) | `lower_from_graph_ir` reconstructing `OnnxModel` from `brain_graph::GraphIr` |
| [`model_zoo`](src/model_zoo.rs) | Reference test architectures: `create_mlp_zoo_model` |
| [`quantize_onnx`](src/quantize_onnx.rs) | `QuantizeOnnxConfig`, `has_quantized_nodes` (Q/DQ operator detection) |
| [`version`](src/version.rs) | `OpsetTable` tracking operator support across opsets 9 through 21 |
| [`testdata`](src/testdata.rs) | `generate_test_op_model` for synthetic fuzzing and operator validation |
| [`tools`](src/tools.rs) | `onnx_summary` generating human-readable model architecture breakdowns |
| [`utils`](src/utils.rs) | LEB128/Varint encoding/decoding, CRC32, and little-endian binary readers |

## Quick Start

```rust
use brain_onnx::{import_and_optimize, ImportConfig, OptimizeConfig};

fn main() {
    let raw_onnx_bytes: &[u8] = b""; // Raw bytes loaded from .onnx file
    let import_cfg = ImportConfig::default();
    let opt_cfg = OptimizeConfig::default();

    let (model, graph_ir) = import_and_optimize(raw_onnx_bytes, &import_cfg, &opt_cfg).unwrap();
    println!("Imported ONNX model: {:?}", model.graph.name);
    println!("Graph IR nodes: {}", graph_ir.num_nodes());
}
```

## Quality & Verification

- **Total Files**: 26 source modules + root `lib.rs`
- **Total Lines of Code**: 83,794 lines
- **Tests**: **9,685 passed · 0 failed · 0 ignored**
- **Clippy**: Clean (`cargo clippy -p brain-onnx -- -D warnings`)
- **Dependencies**: `std` + `brain-core` + `brain-graph` only
