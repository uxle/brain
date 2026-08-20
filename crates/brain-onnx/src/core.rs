//! # ONNX Core Types & Error Model
//!
//! Error enumerations, ONNX version descriptors, and fundamental results.
#![allow(missing_docs)]

use std::fmt;

/// Supported ONNX IR and Opset versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct OnnxVersion {
    pub ir_version: i64,
    pub opset_version: i64,
}

impl OnnxVersion {
    pub const OPSET_9: Self = Self {
        ir_version: 4,
        opset_version: 9,
    };
    pub const OPSET_13: Self = Self {
        ir_version: 7,
        opset_version: 13,
    };
    pub const OPSET_17: Self = Self {
        ir_version: 8,
        opset_version: 17,
    };
    pub const OPSET_21: Self = Self {
        ir_version: 10,
        opset_version: 21,
    };

    pub fn new(ir_version: i64, opset_version: i64) -> Self {
        Self {
            ir_version,
            opset_version,
        }
    }
}

/// Comprehensive error type for ONNX parsing, conversion, and evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum OnnxError {
    ProtobufDecodeError(String),
    UnsupportedOpset(i64),
    UnsupportedOp { op_type: String, domain: String },
    MissingAttribute(String),
    InvalidTensorShape(String),
    GraphLoweringError(String),
    IoError(String),
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxError::ProtobufDecodeError(msg) => write!(f, "Protobuf decode error: {}", msg),
            OnnxError::UnsupportedOpset(v) => write!(f, "Unsupported opset version: {}", v),
            OnnxError::UnsupportedOp { op_type, domain } => {
                write!(f, "Unsupported op: {} (domain: {})", op_type, domain)
            }
            OnnxError::MissingAttribute(name) => write!(f, "Missing required attribute: {}", name),
            OnnxError::InvalidTensorShape(msg) => write!(f, "Invalid tensor shape: {}", msg),
            OnnxError::GraphLoweringError(msg) => write!(f, "Graph lowering error: {}", msg),
            OnnxError::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for OnnxError {}

pub type OnnxResult<T> = Result<T, OnnxError>;

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
