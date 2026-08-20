//! # Activation Quantization & SmoothQuant Scaling
//!
//! Per-token dynamic activation quantization and SmoothQuant outlier migration transforms.
#![allow(
    missing_docs,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

use super::core::{QParams, QuantDType, QuantError, QuantResult, QuantTensor};
use super::utils::{compute_scale_zero_point, minmax, quantize_val};
use brain_core::Tensor;

/// Configuration settings for activation quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct ActQuantConfig {
    pub per_token: bool,
    pub alpha_smoothquant: f64,
    pub dtype: QuantDType,
}

impl Default for ActQuantConfig {
    fn default() -> Self {
        Self {
            per_token: true,
            alpha_smoothquant: 0.5,
            dtype: QuantDType::Int8,
        }
    }
}

/// Activation Quantization Manager.
#[derive(Debug, Clone)]
pub struct ActQuantizer {
    pub config: ActQuantConfig,
}

impl ActQuantizer {
    pub fn new(config: ActQuantConfig) -> Self {
        Self { config }
    }

    /// Computes SmoothQuant migration scale vector.
    pub fn compute_smoothquant_scales(
        &self,
        act_max_per_channel: &[f64],
        weight_max_per_channel: &[f64],
    ) -> QuantResult<Vec<f64>> {
        let n = act_max_per_channel.len();
        if n != weight_max_per_channel.len() || n == 0 {
            return Err(QuantError::ChannelCountMismatch {
                expected: n,
                found: weight_max_per_channel.len(),
            });
        }

        let alpha = self.config.alpha_smoothquant;
        let mut scales = Vec::with_capacity(n);

        for j in 0..n {
            let act_val = act_max_per_channel[j].max(1e-5);
            let weight_val = weight_max_per_channel[j].max(1e-5);
            let s_j = (act_val.powf(alpha) / weight_val.powf(1.0 - alpha)).clamp(1e-5, 1e5);
            scales.push(s_j);
        }

        Ok(scales)
    }

    /// Applies per-token dynamic quantization to 2D activation matrix [Tokens, Channels].
    pub fn quantize_per_token(&self, act: &Tensor) -> QuantResult<QuantTensor> {
        let shape = act.shape();
        if shape.len() != 2 {
            return Err(QuantError::ShapeMismatch {
                expected: vec![1, 1],
                found: shape.to_vec(),
            });
        }

        let num_tokens = shape[0];
        let num_channels = shape[1];
        let data = act.data();

        let mut scales = Vec::with_capacity(num_tokens);
        let mut zero_points = Vec::with_capacity(num_tokens);
        let mut q_data = Vec::with_capacity(num_tokens * num_channels);

        let qmin = self.config.dtype.qmin();
        let qmax = self.config.dtype.qmax();

        for t in 0..num_tokens {
            let start = t * num_channels;
            let end = start + num_channels;
            let slice = &data[start..end];

            let (min_v, max_v) = minmax(slice)?;
            let (scale, zp) = compute_scale_zero_point(min_v, max_v, self.config.dtype, false)?;

            scales.push(scale);
            zero_points.push(zp);

            for &v in slice {
                q_data.push(quantize_val(v, scale, zp, qmin, qmax));
            }
        }

        let params = QParams::per_channel(scales, zero_points, self.config.dtype);
        Ok(QuantTensor::new(q_data, shape.to_vec(), params))
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
