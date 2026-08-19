//! # Fluent Model Export Builder API
//!
//! Fluent configuration builder for multi-format export pipelines.

use crate::core::{ExportFormat, ExportOptions};

/// Fluent builder for export pipelines.
#[derive(Default)]
pub struct ExportBuilder {
    options: ExportOptions,
}

impl ExportBuilder {
    /// Creates a new `ExportBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets target export format.
    pub fn format(mut self, format: ExportFormat) -> Self {
        self.options.format = format;
        self
    }

    /// Sets opset version.
    pub fn opset_version(mut self, version: usize) -> Self {
        self.options.opset_version = version;
        self
    }

    /// Builds the `ExportOptions`.
    pub fn build(self) -> ExportOptions {
        self.options
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
