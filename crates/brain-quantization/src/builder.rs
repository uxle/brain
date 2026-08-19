//! # Fluent Quantization Builder API
//!
//! Declarative configuration builder for constructing complete quantization and pruning pipelines.
#![allow(missing_docs)]

use super::core::{QuantDType, QuantScheme};

/// Supported pipeline modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PipelineMode {
    #[default]
    Dynamic,
    Static,
    QAT,
    Prune,
}

/// Fluent Quantization Pipeline Builder.
#[derive(Debug, Clone)]
pub struct QuantBuilder {
    pub mode: PipelineMode,
    pub dtype: QuantDType,
    pub scheme: QuantScheme,
    pub symmetric: bool,
    pub per_channel: bool,
    pub target_sparsity: f64,
    pub num_calibration_batches: usize,
}

impl Default for QuantBuilder {
    fn default() -> Self {
        Self {
            mode: PipelineMode::Dynamic,
            dtype: QuantDType::Int8,
            scheme: QuantScheme::AffinePerTensor,
            symmetric: false,
            per_channel: false,
            target_sparsity: 0.0,
            num_calibration_batches: 32,
        }
    }
}

impl QuantBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configures dynamic quantization.
    pub fn dynamic_quant(mut self) -> Self {
        self.mode = PipelineMode::Dynamic;
        self
    }

    /// Configures static quantization.
    pub fn static_quant(mut self) -> Self {
        self.mode = PipelineMode::Static;
        self
    }

    /// Configures Quantization-Aware Training (QAT).
    pub fn qat(mut self) -> Self {
        self.mode = PipelineMode::QAT;
        self
    }

    /// Sets target quantized dtype.
    pub fn dtype(mut self, dt: QuantDType) -> Self {
        self.dtype = dt;
        self
    }

    /// Sets target precision to Int8.
    pub fn int8(mut self) -> Self {
        self.dtype = QuantDType::Int8;
        self
    }

    /// Sets target precision to Int4.
    pub fn int4(mut self) -> Self {
        self.dtype = QuantDType::Int4;
        self
    }

    /// Sets symmetric quantization flag.
    pub fn symmetric(mut self, s: bool) -> Self {
        self.symmetric = s;
        self
    }

    /// Sets per-channel quantization flag.
    pub fn per_channel(mut self, pc: bool) -> Self {
        self.per_channel = pc;
        self
    }

    /// Configures target sparsity ratio for pruning.
    pub fn prune_sparsity(mut self, sparsity: f64) -> Self {
        self.target_sparsity = sparsity;
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
