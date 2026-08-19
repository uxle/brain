//! # ONNX Opset Versioning & Compatibility Matrix
//!
//! Tracks supported operator feature sets across ONNX opset versions 9 through 21.
#![allow(missing_docs)]

/// Opset compatibility table lookup.
#[derive(Debug, Clone, Default)]
pub struct OpsetTable;

impl OpsetTable {
    pub fn is_valid_opset(version: i64) -> bool {
        (9..=21).contains(&version)
    }

    pub fn default_opset() -> i64 {
        17
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
