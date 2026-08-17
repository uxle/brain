//! # brain-onnx
//!
//! Production-grade ONNX import, graph optimization, pure-Rust protobuf parser,
//! evaluation interpreter, and Brain Graph IR lowering for the Brain Framework.
//!
//! ## Architecture
//! - [`proto`] — Protobuf wire binary parser for `ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`
//! - [`ir`] — Canonical ONNX Intermediate Representation: `OnnxModel`, `OnnxGraph`, `OnnxNode`, `OnnxValue`
//! - [`import`] — Import pipeline: raw bytes $\to$ `OnnxModel` $\to$ `brain_graph::GraphIr`
//! - [`optimize`] — Graph optimization: constant folding, `Conv+Relu` fusion, `MatMul+Add` ($\to$ Gemm)
//! - [`eval`] — Direct topological evaluation interpreter and `CheckerReport` graph validator
//! - [`export`] — Byte serialization back to standard ONNX binary format
//! - [`ir2graph`] / [`graph2onnx`] — Bidirectional bridge with `brain-graph`
//! - [`model_zoo`] — Tiny fixture reference architectures (MLP, MatMul)
//! - [`version`] — Opset compatibility matrix (opsets 9 through 21)
//! - [`tools`] — `onnx_summary` and operator inventory reporting

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

pub mod config;
pub mod core;
pub mod eval;
pub mod export;
pub mod graph2onnx;
pub mod impl_;
pub mod import;
pub mod ir;
pub mod ir2graph;
pub mod model_zoo;
pub mod ops;
pub mod optimize;
pub mod proto;
pub mod quantize_onnx;
pub mod testdata;
pub mod tools;
pub mod utils;
pub mod version;

// ── Convenience re-exports ──────────────────────────────────────────────────
pub use config::{EvalConfig, ImportConfig, OptimizationLevel, OptimizeConfig, UnknownOpPolicy};
pub use core::{OnnxError, OnnxResult, OnnxVersion};
pub use eval::{check_model, evaluate_onnx_model, CheckerReport};
pub use export::export_onnx_bytes;
pub use graph2onnx::lower_from_graph_ir;
pub use impl_::{import_and_optimize, load_onnx};
pub use import::{import_model, translate_op, ImportReport, UnsupportedOpRegistry};
pub use ir::{OnnxGraph, OnnxModel, OnnxNode, OnnxValue};
pub use ir2graph::lower_to_graph_ir;
pub use model_zoo::create_mlp_zoo_model;
pub use ops::{is_op_supported, OpSpec, STANDARD_OPS};
pub use optimize::{fold_constant_nodes, fuse_conv_relu, fuse_matmul_add, optimize_model};
pub use proto::{
    parse_model_proto, AttributeProto, AttributeType, DataType, GraphProto, ModelProto,
    NodeProto, TensorProto, ValueInfoProto,
};
pub use quantize_onnx::{has_quantized_nodes, QuantizeOnnxConfig};
pub use testdata::generate_test_op_model;
pub use tools::onnx_summary;
pub use utils::{compute_crc32, decode_varint, encode_varint, read_f32_le, read_f64_le};
pub use version::OpsetTable;

/// Framework version string.
pub const VERSION: &str = "0.2.0";