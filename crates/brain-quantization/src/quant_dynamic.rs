//! # Dynamic Quantization Engine
//!
//! Runtime dynamic quantization: statically quantizes model weights and computes runtime activation scales on each forward pass.
#![allow(missing_docs)]

use brain_core::Tensor;
use super::config::DynamicConfig;
use super::core::{QParams, QuantResult, QuantTensor};
use super::utils::{compute_scale_zero_point, minmax, quantize_val};

/// Dynamic Quantization Runner.
#[derive(Debug, Clone)]
pub struct DynamicQuantizer {
    pub config: DynamicConfig,
}

impl DynamicQuantizer {
    pub fn new(config: DynamicConfig) -> Self {
        Self { config }
    }

    /// Quantizes dynamic activation tensor on-the-fly.
    pub fn quantize_activation(&self, activation: &Tensor) -> QuantResult<QuantTensor> {
        let (min_v, max_v) = minmax(activation.data())?;
        let (scale, zp) = compute_scale_zero_point(min_v, max_v, self.config.activation_dtype, false)?;
        let params = QParams::per_tensor(scale, zp, self.config.activation_dtype);

        let qmin = params.qmin;
        let qmax = params.qmax;
        let mut q_data = Vec::with_capacity(activation.numel());

        for &v in activation.data() {
            q_data.push(quantize_val(v, scale, zp, qmin, qmax));
        }

        Ok(QuantTensor::new(q_data, activation.shape().to_vec(), params))
    }

    /// Quantizes weights ahead-of-time per channel or per tensor.
    pub fn quantize_weights(&self, weights: &Tensor) -> QuantResult<QuantTensor> {
        let (min_v, max_v) = minmax(weights.data())?;
        let (scale, zp) = compute_scale_zero_point(min_v, max_v, self.config.weight_dtype, true)?;
        let params = QParams::per_tensor(scale, zp, self.config.weight_dtype);

        let qmin = params.qmin;
        let qmax = params.qmax;
        let mut q_data = Vec::with_capacity(weights.numel());

        for &v in weights.data() {
            q_data.push(quantize_val(v, scale, zp, qmin, qmax));
        }

        Ok(QuantTensor::new(q_data, weights.shape().to_vec(), params))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
