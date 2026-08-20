//! # ONNX Model Import Pipeline
//!
//! Orchestrates byte decoding into `ModelProto` and conversion into canonical `OnnxModel` IR.
#![allow(missing_docs)]

pub mod onnx2graph;
pub mod ops;
pub mod unsupported;

pub use onnx2graph::proto_to_ir;
pub use ops::translate_op;
pub use unsupported::{UnsupportedOpRegistry, UnsupportedReport};

use crate::config::ImportConfig;
use crate::core::OnnxResult;
use crate::ir::OnnxModel;
use crate::proto::parse_model_proto;

/// Summary report returned after importing an ONNX model.
#[derive(Debug, Clone, Default)]
pub struct ImportReport {
    pub total_nodes: usize,
    pub total_initializers: usize,
    pub unsupported_ops: Vec<String>,
}

/// Imports raw bytes into canonical OnnxModel IR.
pub fn import_model(bytes: &[u8], config: &ImportConfig) -> OnnxResult<OnnxModel> {
    let proto = parse_model_proto(bytes)?;
    proto_to_ir(&proto, config)
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
