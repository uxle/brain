//! # Supported Operation Tables
//!
//! Format compatibility matrices and op-level status checkers.

use crate::core::ExportFormat;

/// Checks if a primitive operator is supported by a given export format.
pub fn is_op_supported(op_name: &str, _format: ExportFormat) -> bool {
    matches!(
        op_name,
        "Add" | "Sub" | "Mul" | "Div" | "MatMul" | "Conv2d" | "Relu" | "Softmax"
    )
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
