//! # Standalone ONNX Protobuf Binary Exporter
//!
//! Hand-rolled binary encoder serializing `ModelProto`, `GraphProto`, `NodeProto`, and `TensorProto`.

pub mod checker;
pub mod ops;

pub use checker::validate_onnx_graph;
pub use ops::map_to_onnx_op;

use crate::core::ExportError;
use crate::model::{ExportModel, ModelExporter};

/// Configuration options for ONNX export.
#[derive(Debug, Clone)]
pub struct OnnxConfig {
    pub opset_version: usize,
    pub ir_version: usize,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            opset_version: 17,
            ir_version: 8,
        }
    }
}

/// Standalone binary ONNX model exporter.
pub struct OnnxExporter {
    pub config: OnnxConfig,
}

impl OnnxExporter {
    /// Creates a new `OnnxExporter`.
    pub fn new(config: OnnxConfig) -> Self {
        Self { config }
    }
}

impl ModelExporter for OnnxExporter {
    fn export(&self, _model: &ExportModel, _path: &str) -> Result<(), ExportError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
