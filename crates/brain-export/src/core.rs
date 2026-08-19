//! # Core Model Export Types
//!
//! Provides the primary [`ExportFormat`], [`ExportOptions`], and [`ExportError`] definitions.

/// Target neural network model deployment formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportFormat {
    #[default]
    Onnx,
    Tflite,
    CoreMl,
    WebNn,
}

/// Common options configuring export precision, opset, and verification.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    pub format: ExportFormat,
    pub opset_version: usize,
    pub quantize: bool,
    pub verify: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::default(),
            opset_version: 17,
            quantize: false,
            verify: true,
        }
    }
}

/// Errors occurring during model export or serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportError {
    UnsupportedOp(String),
    SerializationError(String),
    VerificationFailed(String),
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
