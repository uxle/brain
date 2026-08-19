//! # W3C WebNN Graph Descriptor Exporter
//!
//! Serializes computational graphs into JSON-compatible operand representations for WebNN browser runtimes.

pub mod ops;

use crate::core::ExportError;
use crate::model::{ExportModel, ModelExporter};

/// WebNN export configuration.
#[derive(Debug, Clone, Default)]
pub struct WebnnConfig;

/// WebNN graph exporter.
pub struct WebnnExporter {
    pub config: WebnnConfig,
}

impl WebnnExporter {
    /// Creates a new `WebnnExporter`.
    pub fn new(config: WebnnConfig) -> Self {
        Self { config }
    }
}

impl ModelExporter for WebnnExporter {
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
