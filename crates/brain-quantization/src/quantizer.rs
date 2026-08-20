//! # Quantizer Abstractions & Engines
//!
//! Affine, symmetric, and per-channel quantizers transforming continuous tensors to discrete representations.
#![allow(missing_docs)]

use super::core::{QParams, QuantDType, QuantResult, QuantScheme, QuantTensor};
use super::utils::{compute_scale_zero_point, minmax, quantize_val};
use brain_core::Tensor;

/// Architectural kind of quantizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizerKind {
    #[default]
    AffinePerTensor,
    SymmetricPerTensor,
    AffinePerChannel,
    SymmetricPerChannel,
}

/// Fundamental trait implemented by tensor quantizers.
pub trait Quantizer: Send + Sync {
    /// Quantizes a continuous floating point tensor into integer QuantTensor.
    fn quantize(&self, tensor: &Tensor) -> QuantResult<QuantTensor>;

    /// Dequantizes a QuantTensor back into a floating point Tensor.
    fn dequantize(&self, qtensor: &QuantTensor) -> QuantResult<Tensor>;

    /// Returns current quantization parameters.
    fn get_params(&self) -> &QParams;
}

/// Affine Per-Tensor Quantizer.
#[derive(Debug, Clone)]
pub struct AffineQuantizer {
    pub params: QParams,
}

impl AffineQuantizer {
    pub fn new(scale: f64, zero_point: i32, dtype: QuantDType) -> Self {
        Self {
            params: QParams::per_tensor(scale, zero_point, dtype),
        }
    }

    pub fn from_tensor(tensor: &Tensor, dtype: QuantDType) -> QuantResult<Self> {
        let (min_v, max_v) = minmax(tensor.data())?;
        let (scale, zp) = compute_scale_zero_point(min_v, max_v, dtype, false)?;
        Ok(Self::new(scale, zp, dtype))
    }
}

impl Quantizer for AffineQuantizer {
    fn quantize(&self, tensor: &Tensor) -> QuantResult<QuantTensor> {
        let scale = self.params.scales[0];
        let zp = self.params.zero_points[0];
        let qmin = self.params.qmin;
        let qmax = self.params.qmax;

        let mut q_data = Vec::with_capacity(tensor.numel());
        for &v in tensor.data() {
            q_data.push(quantize_val(v, scale, zp, qmin, qmax));
        }

        Ok(QuantTensor::new(
            q_data,
            tensor.shape().to_vec(),
            self.params.clone(),
        ))
    }

    fn dequantize(&self, qtensor: &QuantTensor) -> QuantResult<Tensor> {
        Ok(qtensor.dequantize())
    }

    fn get_params(&self) -> &QParams {
        &self.params
    }
}

/// Symmetric Per-Tensor Quantizer (zero-point = 0).
#[derive(Debug, Clone)]
pub struct SymmetricQuantizer {
    pub params: QParams,
}

impl SymmetricQuantizer {
    pub fn new(scale: f64, dtype: QuantDType) -> Self {
        let mut params = QParams::per_tensor(scale, 0, dtype);
        params.scheme = QuantScheme::SymmetricPerTensor;
        Self { params }
    }

    pub fn from_tensor(tensor: &Tensor, dtype: QuantDType) -> QuantResult<Self> {
        let (min_v, max_v) = minmax(tensor.data())?;
        let (scale, _) = compute_scale_zero_point(min_v, max_v, dtype, true)?;
        Ok(Self::new(scale, dtype))
    }
}

impl Quantizer for SymmetricQuantizer {
    fn quantize(&self, tensor: &Tensor) -> QuantResult<QuantTensor> {
        let scale = self.params.scales[0];
        let qmin = self.params.qmin;
        let qmax = self.params.qmax;

        let mut q_data = Vec::with_capacity(tensor.numel());
        for &v in tensor.data() {
            q_data.push(quantize_val(v, scale, 0, qmin, qmax));
        }

        Ok(QuantTensor::new(
            q_data,
            tensor.shape().to_vec(),
            self.params.clone(),
        ))
    }

    fn dequantize(&self, qtensor: &QuantTensor) -> QuantResult<Tensor> {
        Ok(qtensor.dequantize())
    }

    fn get_params(&self) -> &QParams {
        &self.params
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
