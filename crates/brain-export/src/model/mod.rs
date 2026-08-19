//! # Universal Export Model Abstraction
//!
//! Represents neural network models as exported parameter collections and computational graphs.

use crate::core::ExportError;
use brain_core::Tensor;

/// Abstract neural network model for export.
#[derive(Debug, Clone)]
pub struct ExportModel {
    pub name: String,
    pub parameters: Vec<(String, Tensor)>,
}

impl ExportModel {
    /// Creates a new `ExportModel`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parameters: Vec::new(),
        }
    }

    /// Adds a named weight parameter tensor.
    pub fn add_parameter(&mut self, name: impl Into<String>, tensor: Tensor) {
        self.parameters.push((name.into(), tensor));
    }
}

/// Exporter interface for converting models to target file formats.
pub trait ModelExporter: Send + Sync {
    fn export(&self, model: &ExportModel, path: &str) -> Result<(), ExportError>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
