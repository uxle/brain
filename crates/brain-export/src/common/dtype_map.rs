//! # Data Type Mapping Tables
//!
//! Translates internal data types to ONNX, TFLite, CoreML, and WebNN type identifiers.

/// Standard tensor data type tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DTypeKind {
    #[default]
    Float32,
    Float64,
    Int32,
    Int64,
}

/// Maps standard `DTypeKind` to ONNX tensor proto data type integer.
pub fn map_dtype_to_onnx(dtype: DTypeKind) -> i32 {
    match dtype {
        DTypeKind::Float32 => 1,
        DTypeKind::Float64 => 11,
        DTypeKind::Int32 => 6,
        DTypeKind::Int64 => 7,
    }
}

/// Maps standard `DTypeKind` to TFLite tensor type integer.
pub fn map_dtype_to_tflite(dtype: DTypeKind) -> i32 {
    match dtype {
        DTypeKind::Float32 => 0,
        DTypeKind::Float64 => 8,
        DTypeKind::Int32 => 2,
        DTypeKind::Int64 => 4,
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
