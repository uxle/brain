//! # Supported Operator Compatibility Registry
//!
//! Queryable registry of operator support across all export formats.

/// Supported operations audit report.
#[derive(Debug, Clone, Default)]
pub struct SupportedOpsReport {
    pub supported_count: usize,
}

impl SupportedOpsReport {
    /// Creates a new `SupportedOpsReport`.
    pub fn new(count: usize) -> Self {
        Self {
            supported_count: count,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
