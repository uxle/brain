//! # Quantization & Pruning Functional Implementations
//!
//! High-level APIs for converting models, tensors, and layers to quantized representations.
#![allow(missing_docs)]

use brain_core::Tensor;
use super::config::QuantConfig;
use super::core::{QParams, QuantResult, QuantScheme, QuantTensor};
use super::prune::{MagnitudePruner, PruneResult, Pruner};
use super::quantizer::{AffineQuantizer, Quantizer, SymmetricQuantizer};
use super::utils::{compute_scale_zero_point, minmax, quantize_val};

/// Quantizes an arbitrary floating point Tensor according to configuration.
pub fn quantize_tensor(tensor: &Tensor, config: &QuantConfig) -> QuantResult<QuantTensor> {
    if config.per_channel
        || matches!(
            config.scheme,
            QuantScheme::AffinePerChannel | QuantScheme::SymmetricPerChannel
        )
    {
        let num_channels = tensor.shape().first().copied().unwrap_or(1);
        let channel_size = tensor.numel().checked_div(num_channels).unwrap_or(0);
        let is_sym = config.symmetric
            || matches!(
                config.scheme,
                QuantScheme::SymmetricPerChannel | QuantScheme::SymmetricPerTensor
            );

        let data = tensor.data();
        let mut scales = Vec::with_capacity(num_channels);
        let mut zero_points = Vec::with_capacity(num_channels);

        for ch in 0..num_channels {
            let start = ch * channel_size;
            let end = if ch == num_channels - 1 {
                data.len()
            } else {
                (start + channel_size).min(data.len())
            };
            let (min_v, max_v) = minmax(&data[start..end])?;
            let (scale, zp) = compute_scale_zero_point(min_v, max_v, config.dtype, is_sym)?;
            scales.push(scale);
            zero_points.push(if is_sym { 0 } else { zp });
        }

        let mut params = QParams::per_channel(scales, zero_points, config.dtype);
        if is_sym {
            params.scheme = QuantScheme::SymmetricPerChannel;
        } else {
            params.scheme = QuantScheme::AffinePerChannel;
        }

        let mut q_data = Vec::with_capacity(tensor.numel());
        let qmin = params.qmin;
        let qmax = params.qmax;
        for ch in 0..num_channels {
            let scale = params.scales[ch];
            let zp = params.zero_points[ch];
            let start = ch * channel_size;
            let end = if ch == num_channels - 1 {
                data.len()
            } else {
                (start + channel_size).min(data.len())
            };
            for &val in &data[start..end] {
                q_data.push(quantize_val(val, scale, zp, qmin, qmax));
            }
        }
        Ok(QuantTensor::new(q_data, tensor.shape().to_vec(), params))
    } else if config.symmetric || matches!(config.scheme, QuantScheme::SymmetricPerTensor) {
        let quantizer = SymmetricQuantizer::from_tensor(tensor, config.dtype)?;
        quantizer.quantize(tensor)
    } else {
        let quantizer = AffineQuantizer::from_tensor(tensor, config.dtype)?;
        quantizer.quantize(tensor)
    }
}

/// Dequantizes a QuantTensor back to full precision floating point Tensor.
pub fn dequantize_tensor(qtensor: &QuantTensor) -> QuantResult<Tensor> {
    Ok(qtensor.dequantize())
}

/// Applies unstructured magnitude pruning to target tensor.
pub fn apply_magnitude_prune(weights: &mut Tensor, sparsity: f64) -> QuantResult<PruneResult> {
    let pruner = MagnitudePruner::new(sparsity);
    pruner.prune_in_place(weights)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
