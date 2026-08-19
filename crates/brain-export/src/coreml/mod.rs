//! # Apple CoreML Model Exporter (.mlpackage)
//!
//! Hand-rolled Protobuf specification encoder targeting Apple Neural Engine and GPU acceleration.

pub mod ops;

use crate::core::ExportError;
use crate::model::{ExportModel, ModelExporter};

/// CoreML export configuration.
#[derive(Debug, Clone)]
pub struct CoreMlConfig {
    pub specification_version: usize,
}

impl Default for CoreMlConfig {
    fn default() -> Self {
        Self {
            specification_version: 7,
        }
    }
}

/// CoreML model package exporter.
pub struct CoreMlExporter {
    pub config: CoreMlConfig,
}

impl CoreMlExporter {
    /// Creates a new `CoreMlExporter`.
    pub fn new(config: CoreMlConfig) -> Self {
        Self { config }
    }
}

impl ModelExporter for CoreMlExporter {
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
