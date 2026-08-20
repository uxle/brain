//! # Fake Quantization & Quantization-Aware Training (QAT)
//!
//! Simulates quantization noise in the forward pass using Straight-Through Estimator (STE) gradient propagation.
#![allow(missing_docs)]

use super::config::FakeQuantConfig;
use super::core::QuantResult;
use super::utils::{compute_scale_zero_point, dequantize_val, minmax, quantize_val};
use brain_core::Tensor;

/// Fake Quantization Module for QAT.
#[derive(Debug, Clone)]
pub struct FakeQuantize {
    pub config: FakeQuantConfig,
    pub scale: f64,
    pub zero_point: i32,
    pub is_enabled: bool,
}

impl FakeQuantize {
    pub fn new(config: FakeQuantConfig) -> Self {
        Self {
            config,
            scale: 1.0,
            zero_point: 0,
            is_enabled: true,
        }
    }

    /// Initializes fake quant parameters from sample tensor min/max.
    pub fn init_from_tensor(&mut self, tensor: &Tensor) -> QuantResult<()> {
        let (min_v, max_v) = minmax(tensor.data())?;
        let (scale, zp) =
            compute_scale_zero_point(min_v, max_v, self.config.dtype, self.config.symmetric)?;
        self.scale = scale;
        self.zero_point = zp;
        Ok(())
    }

    /// Forward pass: simulates clamp round scale.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        if !self.is_enabled {
            return input.clone();
        }

        let qmin = self.config.dtype.qmin();
        let qmax = self.config.dtype.qmax();
        let scale = self.scale;
        let zp = self.zero_point;

        let mut out_data = Vec::with_capacity(input.numel());
        for &v in input.data() {
            let q = quantize_val(v, scale, zp, qmin, qmax);
            out_data.push(dequantize_val(q, scale, zp));
        }

        Tensor::from_slice(&out_data, input.shape().to_vec())
    }

    /// Backward pass (STE): passes gradient through if input within [qmin*scale, qmax*scale].
    pub fn backward_ste(&self, grad_output: &Tensor, input: &Tensor) -> Tensor {
        if !self.config.ste_grad_clip {
            return grad_output.clone();
        }

        let qmin = self.config.dtype.qmin() as f64;
        let qmax = self.config.dtype.qmax() as f64;
        let min_bound = (qmin - self.zero_point as f64) * self.scale;
        let max_bound = (qmax - self.zero_point as f64) * self.scale;

        let g_data = grad_output.data();
        let in_data = input.data();
        let n = g_data.len();
        let mut g_in = vec![0.0; n];

        for i in 0..n {
            let x = in_data[i];
            if x >= min_bound && x <= max_bound {
                g_in[i] = g_data[i];
            } else {
                g_in[i] = 0.0;
            }
        }

        Tensor::from_slice(&g_in, grad_output.shape().to_vec())
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
