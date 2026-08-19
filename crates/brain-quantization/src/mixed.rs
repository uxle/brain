//! # Mixed Precision Quantization
//!
//! Layer-wise sensitivity estimation and automatic bit-width allocation (int4 vs int8 vs fp16).
#![allow(missing_docs)]

use std::collections::HashMap;
use super::core::QuantDType;

/// Configuration container for mixed-precision selection.
#[derive(Debug, Clone, PartialEq)]
pub struct MixedConfig {
    pub default_dtype: QuantDType,
    pub sensitive_threshold: f64,
}

impl Default for MixedConfig {
    fn default() -> Self {
        Self {
            default_dtype: QuantDType::Int8,
            sensitive_threshold: 0.1,
        }
    }
}

/// Mixed Precision Quantizer allocating bit-widths based on layer perturbation sensitivity.
#[derive(Debug, Clone)]
pub struct MixedPrecisionQuantizer {
    pub config: MixedConfig,
    pub layer_sensitivities: HashMap<String, f64>,
}

impl MixedPrecisionQuantizer {
    pub fn new(config: MixedConfig) -> Self {
        Self {
            config,
            layer_sensitivities: HashMap::new(),
        }
    }

    /// Registers observed loss perturbation sensitivity for a named model layer.
    pub fn register_sensitivity(&mut self, layer_name: impl Into<String>, sensitivity: f64) {
        self.layer_sensitivities.insert(layer_name.into(), sensitivity);
    }

    /// Determines the optimal precision type for a given layer based on sensitivity.
    pub fn select_dtype_for_layer(&self, layer_name: &str) -> QuantDType {
        if let Some(&sens) = self.layer_sensitivities.get(layer_name) {
            if sens > self.config.sensitive_threshold * 5.0 {
                QuantDType::Float16
            } else if sens > self.config.sensitive_threshold {
                QuantDType::Int8
            } else {
                QuantDType::Int4
            }
        } else {
            self.config.default_dtype
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
