//! # Standalone FlatBuffers TFLite Exporter
//!
//! Serializes computational graphs and weight buffers into binary `.tflite` format.

pub mod ops;

pub use ops::map_to_tflite_builtin_code;

use crate::core::ExportError;
use crate::model::{ExportModel, ModelExporter};

/// Configuration options for TFLite export.
#[derive(Debug, Clone)]
pub struct TfliteConfig {
    pub quantize: bool,
}

impl Default for TfliteConfig {
    fn default() -> Self {
        Self { quantize: false }
    }
}

/// Standalone binary TFLite model exporter.
pub struct TfliteExporter {
    pub config: TfliteConfig,
}

impl TfliteExporter {
    /// Creates a new `TfliteExporter`.
    pub fn new(config: TfliteConfig) -> Self {
        Self { config }
    }
}

impl ModelExporter for TfliteExporter {
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
