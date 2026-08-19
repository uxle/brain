//! # Unsupported Operator Diagnostics & Registry
//!
//! Diagnostic tracking and policy enforcement for non-standard or unsupported ONNX operators.
#![allow(missing_docs)]

use std::collections::HashSet;

/// Registry tracking unsupported operators encountered during import.
#[derive(Debug, Clone, Default)]
pub struct UnsupportedOpRegistry {
    pub unsupported_ops: HashSet<String>,
}

impl UnsupportedOpRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_unsupported(&mut self, op_name: impl Into<String>) {
        self.unsupported_ops.insert(op_name.into());
    }

    pub fn is_empty(&self) -> bool {
        self.unsupported_ops.is_empty()
    }
}

/// Diagnostic report summarizing unsupported operators.
#[derive(Debug, Clone, Default)]
pub struct UnsupportedReport {
    pub missing_ops: Vec<String>,
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
