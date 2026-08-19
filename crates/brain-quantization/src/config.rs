//! # Quantization Configuration Presets
//!
//! Comprehensive settings for static/dynamic quantization, QAT, pruning, and sparse algebra.
#![allow(missing_docs)]

use super::core::{QuantDType, QuantScheme};

/// General quantization configuration container.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantConfig {
    pub dtype: QuantDType,
    pub scheme: QuantScheme,
    pub symmetric: bool,
    pub per_channel: bool,
    pub preserve_zero: bool,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            dtype: QuantDType::Int8,
            scheme: QuantScheme::AffinePerTensor,
            symmetric: false,
            per_channel: false,
            preserve_zero: true,
        }
    }
}

/// Configuration settings for dynamic quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicConfig {
    pub weight_dtype: QuantDType,
    pub activation_dtype: QuantDType,
    pub per_channel_weights: bool,
}

impl Default for DynamicConfig {
    fn default() -> Self {
        Self {
            weight_dtype: QuantDType::Int8,
            activation_dtype: QuantDType::UInt8,
            per_channel_weights: true,
        }
    }
}

/// Configuration settings for static offline quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticConfig {
    pub weight_dtype: QuantDType,
    pub activation_dtype: QuantDType,
    pub num_calibration_batches: usize,
    pub per_channel_weights: bool,
}

impl Default for StaticConfig {
    fn default() -> Self {
        Self {
            weight_dtype: QuantDType::Int8,
            activation_dtype: QuantDType::UInt8,
            num_calibration_batches: 32,
            per_channel_weights: true,
        }
    }
}

/// Configuration settings for fake quantization and Quantization-Aware Training (QAT).
#[derive(Debug, Clone, PartialEq)]
pub struct FakeQuantConfig {
    pub dtype: QuantDType,
    pub symmetric: bool,
    pub per_channel: bool,
    pub ste_grad_clip: bool,
}

impl Default for FakeQuantConfig {
    fn default() -> Self {
        Self {
            dtype: QuantDType::Int8,
            symmetric: false,
            per_channel: false,
            ste_grad_clip: true,
        }
    }
}

/// Configuration settings for model pruning.
#[derive(Debug, Clone, PartialEq)]
pub struct PruneConfig {
    pub target_sparsity: f64,
    pub structured: bool,
    pub channel_axis: usize,
    pub preserve_norm: bool,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            target_sparsity: 0.5,
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        }
    }
}

/// Configuration settings for block/group-wise quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQuantConfig {
    pub group_size: usize,
    pub dtype: QuantDType,
    pub symmetric: bool,
}

impl Default for BlockQuantConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            dtype: QuantDType::Int4,
            symmetric: true,
        }
    }
}

/// Configuration settings for sparse matrix computation.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseConfig {
    pub zero_threshold: f64,
    pub format: SparseFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SparseFormat {
    #[default]
    Csr,
    Csc,
    Coo,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            zero_threshold: 1e-6,
            format: SparseFormat::Csr,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
